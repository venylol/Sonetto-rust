#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
import sys
import re
from dataclasses import dataclass

@dataclass
class Stats:
    games: int = 0
    eg_w: int = 0
    eg_l: int = 0
    d: int = 0
    eg_disc_diff_sum: int = 0  # (Egaroucid - Sensei)
    eg_as_black_w: int = 0
    eg_as_black_l: int = 0
    eg_as_black_d: int = 0
    eg_as_white_w: int = 0
    eg_as_white_l: int = 0
    eg_as_white_d: int = 0

def winrate(w, d, g):
    return (w + 0.5 * d) / g if g else 0.0

def parse_final_score(text: str) -> int:
    s = (text or "").strip()
    if s == "0":
        return 0
    if len(s) >= 3 and s[1] == "+":
        sign = 1 if s[0] in ("B", "b") else -1
        try:
            return sign * int(s[2:])
        except Exception:
            pass
    m = re.search(r"Final score is B\s+(\d+)\s+and W\s+(\d+)", s)
    if m:
        b = int(m.group(1)); w = int(m.group(2))
        return b - w
    return 0

class GtpProc:
    def __init__(self, cmd, cwd=None, extra_env=None):
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        self.p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._id = 1

    def send(self, command: str) -> str:
        cid = self._id
        self._id += 1
        self.p.stdin.write(f"{cid} {command}\n")
        self.p.stdin.flush()
        # Egaroucid console prints board/diagnostics to stdout. We ignore any lines until we see
        # the GTP response marker ('=' or '?') for our command id, then read until the blank terminator.
        started = False
        resp_lines = []
        while True:
            line = self.p.stdout.readline()
            if line == "":
                err = ""
                if self.p.stderr:
                    try:
                        err = self.p.stderr.read()
                    except Exception:
                        err = ""
                raise RuntimeError(f"engine terminated. stderr={err[:500]}")
            line = line.rstrip("\n")
            if not started:
                if not (line.startswith("=") or line.startswith("?")):
                    continue
                # If engine echoes id, require id match. Otherwise accept first marker line.
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    if int(parts[1]) != cid:
                        continue
                started = True
                resp_lines.append(line)
                continue
            # started
            if line == "":
                break
            resp_lines.append(line)
        return "\n".join(resp_lines)
    def quit(self):
        try:
            self.send("quit")
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass

def xot_to_opening_line(line: str):
    line = line.strip()
    coords = [line[i:i+2] for i in range(0, len(line), 2)]
    moves = []
    for j, c in enumerate(coords):
        moves.append(("b" if (j % 2 == 0) else "w", c))
    return moves

def apply_opening(engine: GtpProc, opening_moves):
    engine.send("clear_board")
    for color, coord in opening_moves:
        resp = engine.send(f"play {color} {coord}")
        if resp.startswith("?"):
            raise RuntimeError(f"illegal opening move {color} {coord}: {resp}")

def genmove(engine: GtpProc, color: str) -> str:
    resp = engine.send(f"genmove {color}")
    if resp.startswith("?"):
        raise RuntimeError(f"genmove error: {resp}")
    parts = resp.split()
    return parts[-1] if parts else "PASS"

def final_result_egar(egar: GtpProc) -> str:
    resp = egar.send("gogui-rules_final_result")
    if resp.startswith("?"):
        return ""
    return resp

def play_one_game(egar: GtpProc, sensei: GtpProc, opening_moves, egar_is_black: bool):
    apply_opening(egar, opening_moves)
    apply_opening(sensei, opening_moves)

    stm_black = (len(opening_moves) % 2 == 0)
    pass_count = 0

    for _ply in range(120):
        if pass_count >= 2:
            break

        color = "b" if stm_black else "w"

        if (color == "b" and egar_is_black) or (color == "w" and not egar_is_black):
            move = genmove(egar, color)
            if move.upper() == "PASS":
                # Egaroucid doesn't update internal board on PASS in genmove
                resp0 = egar.send(f"play {color} PASS")
                if resp0.startswith("?"):
                    raise RuntimeError(f"egaroucid rejected PASS {color}: {resp0}")
            resp2 = sensei.send(f"play {color} {move}")
            if resp2.startswith("?"):
                raise RuntimeError(f"sensei rejected move {color} {move}: {resp2}")
        else:
            move = genmove(sensei, color)
            resp2 = egar.send(f"play {color} {move}")
            if resp2.startswith("?"):
                raise RuntimeError(f"egaroucid rejected move {color} {move}: {resp2}")

        pass_count = (pass_count + 1) if move.upper() == "PASS" else 0
        stm_black = not stm_black
    else:
        raise RuntimeError("ply limit exceeded (possible desync); check play errors above")

    fr = final_result_egar(egar)
    diff_bw = parse_final_score(fr)
    eg_diff = diff_bw if egar_is_black else -diff_bw

    if eg_diff > 0:
        outcome = "W"
    elif eg_diff < 0:
        outcome = "L"
    else:
        outcome = "D"
    return outcome, eg_diff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--report_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--xot_file", type=str, default="XOT opening.txt")
    ap.add_argument("--egar", type=str, default="./build/egaroucid/egaroucid_gtp")
    ap.add_argument("--sensei", type=str, default="./build/sensei/sensei_depth1_gtp")
    ap.add_argument("--log", type=str, default="benchmark_results.log")
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    with open(args.xot_file, "r", encoding="utf-8", errors="ignore") as f:
        xot_lines = [ln.strip() for ln in f if ln.strip()]
    if not xot_lines:
        print("XOT opening file empty", file=sys.stderr)
        sys.exit(2)

    egar = GtpProc([args.egar, "-gtp", "-q"])
    sensei = GtpProc([args.sensei, "--eval", "sensei-engine/pattern_evaluator.dat"])

    st = Stats()

    with open(args.log, "w", encoding="utf-8") as out:
        for g in range(args.games):
            opening = xot_to_opening_line(rnd.choice(xot_lines))
            egar_is_black = (g % 2 == 0)

            outcome, eg_diff = play_one_game(egar, sensei, opening, egar_is_black)

            st.games += 1
            st.eg_disc_diff_sum += eg_diff

            if egar_is_black:
                if outcome == "W": st.eg_as_black_w += 1
                elif outcome == "L": st.eg_as_black_l += 1
                else: st.eg_as_black_d += 1
            else:
                if outcome == "W": st.eg_as_white_w += 1
                elif outcome == "L": st.eg_as_white_l += 1
                else: st.eg_as_white_d += 1

            if outcome == "W":
                st.eg_w += 1
            elif outcome == "L":
                st.eg_l += 1
            else:
                st.d += 1

            if st.games % args.report_every == 0:
                wr = winrate(st.eg_w, st.d, st.games) * 100.0
                avgdiff = st.eg_disc_diff_sum / st.games
                msg = (
                    f"Games: {st.games}\n"
                    f"Egaroucid: {st.eg_w}W {st.eg_l}L {st.d}D  (WinRate={wr:.2f}%)\n"
                    f"Sensei   : {st.eg_l}W {st.eg_w}L {st.d}D\n"
                    f"Avg disc diff (Egaroucid - Sensei): {avgdiff:.3f}\n"
                    f"As Black: {st.eg_as_black_w}W {st.eg_as_black_l}L {st.eg_as_black_d}D\n"
                    f"As White: {st.eg_as_white_w}W {st.eg_as_white_l}L {st.eg_as_white_d}D\n"
                    f"---\n"
                )
                print(msg, end="")
                out.write(msg)
                out.flush()

    egar.quit()
    sensei.quit()

if __name__ == "__main__":
    main()
