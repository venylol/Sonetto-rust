
#include <iostream>
#include <sstream>
#include <string>
#include <cctype>
#include <algorithm>

#include "board/board.h"
#include "board/get_moves.h"
#include "board/get_flip.h"
#include "evaluatedepthone/pattern_evaluator.h"

using namespace std;

static inline string Ok(int id, const string& payload = "") {
  if (id >= 0) {
    if (payload.empty()) return "= " + to_string(id) + "\n\n";
    return "= " + to_string(id) + " " + payload + "\n\n";
  }
  if (payload.empty()) return "=\n\n";
  return "= " + payload + "\n\n";
}
static inline string Err(int id, const string& payload) {
  if (id >= 0) return "? " + to_string(id) + " " + payload + "\n\n";
  return "? " + payload + "\n\n";
}

static inline bool ParseColor(const string& s, bool* is_black) {
  string t; t.reserve(s.size());
  for (char c : s) t.push_back((char)tolower((unsigned char)c));
  if (t=="b" || t=="black") { *is_black = true; return true; }
  if (t=="w" || t=="white") { *is_black = false; return true; }
  return false;
}

static inline bool ParseCoord(const string& s, Square* out_sq, bool* is_pass) {
  if (s=="PASS" || s=="pass") { *is_pass = true; return true; }
  if (s.size()!=2) return false;
  char file = (char)tolower((unsigned char)s[0]);
  char rank = s[1];
  if (file<'a' || file>'h') return false;
  if (rank<'1' || rank>'8') return false;
  int x = file - 'a';
  int y = rank - '1';
  *out_sq = (Square)(y*8 + x); // a1=0 ... h8=63
  *is_pass = false;
  return true;
}

static inline string SqToCoord(Square sq) {
  int x = (int)sq % 8;
  int y = (int)sq / 8;
  string r;
  r.push_back((char)('a' + x));
  r.push_back((char)('1' + y));
  return r;
}

struct GameState {
  Board board;
  bool stm_is_black = true;
};

static inline void Reset(GameState& st) {
  st.board = Board();
  st.stm_is_black = true;
}

static inline void ApplyPass(GameState& st) {
  st.board.PlayMove(0);
  st.stm_is_black = !st.stm_is_black;
}

static inline bool ApplyMove(GameState& st, Square mv) {
  BitPattern moves = GetMoves(st.board.Player(), st.board.Opponent());
  if (((moves >> (int)mv) & 1ULL) == 0ULL) return false;
  BitPattern flip = GetFlip(mv, st.board.Player(), st.board.Opponent());
  st.board.PlayMove(flip);
  st.stm_is_black = !st.stm_is_black;
  return true;
}

static inline bool IsOver(const GameState& st) { return IsGameOver(st.board); }

static inline pair<int,int> CountDiscsAbs(const GameState& st) {
  BitPattern player = st.board.Player();
  BitPattern opp = st.board.Opponent();
  BitPattern black = st.stm_is_black ? player : opp;
  BitPattern white = st.stm_is_black ? opp : player;
  return {__builtin_popcountll(black), __builtin_popcountll(white)};
}

int main(int argc, char** argv) {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  string eval_path = "sensei-engine/pattern_evaluator.dat";
  for (int i=1;i<argc;i++) {
    string a = argv[i];
    if (a=="--eval" && i+1<argc) eval_path = argv[++i];
  }

  auto evals = LoadEvals(eval_path);
  if (evals.empty()) evals = LoadEvals("pattern_evaluator.dat");

  GameState st;
  Reset(st);

  string line;
  while (getline(cin, line)) {
    if (line.empty()) continue;

    int id = -1;
    string cmdline = line;
    {
      size_t p = cmdline.find_first_not_of(" \t\r\n");
      if (p!=string::npos) cmdline = cmdline.substr(p);
    }

    // optional numeric id
    {
      stringstream ss(cmdline);
      string tok;
      ss >> tok;
      bool all_digits = !tok.empty() && all_of(tok.begin(), tok.end(),
                        [](char c){ return isdigit((unsigned char)c); });
      if (all_digits) {
        id = stoi(tok);
        string rest;
        getline(ss, rest);
        cmdline = rest;
        size_t p = cmdline.find_first_not_of(" \t\r\n");
        if (p!=string::npos) cmdline = cmdline.substr(p);
        else cmdline.clear();
      }
    }

    stringstream iss(cmdline);
    string cmd;
    iss >> cmd;
    if (cmd.empty()) continue;
    for (auto& c : cmd) c = (char)tolower((unsigned char)c);

    if (cmd=="quit" || cmd=="exit") {
      cout << Ok(id) << flush;
      break;
    }
    if (cmd=="clear_board") {
      Reset(st);
      cout << Ok(id) << flush;
      continue;
    }
    if (cmd=="play") {
      string color_s, coord_s;
      iss >> color_s >> coord_s;
      bool is_black=false;
      if (!ParseColor(color_s, &is_black)) { cout << Err(id,"illegal color") << flush; continue; }
      Square sq = 0; bool is_pass=false;
      if (!ParseCoord(coord_s, &sq, &is_pass)) { cout << Err(id,"illegal move") << flush; continue; }

      if (is_black != st.stm_is_black) { cout << Err(id, "wrong color to play") << flush; continue; }
      if (is_pass) { ApplyPass(st); cout << Ok(id) << flush; continue; }
      if (!ApplyMove(st, sq)) { cout << Err(id,"illegal move") << flush; continue; }
      cout << Ok(id) << flush;
      continue;
    }
    if (cmd=="genmove") {
      string color_s; iss >> color_s;
      bool is_black=false;
      if (!ParseColor(color_s, &is_black)) { cout << Err(id,"illegal color") << flush; continue; }
      if (evals.empty()) { cout << Err(id,"evals not loaded") << flush; continue; }

      if (is_black != st.stm_is_black) { cout << Err(id, "wrong color to play") << flush; continue; }

      BitPattern moves = GetMoves(st.board.Player(), st.board.Opponent());
      if (moves == 0ULL) {
        ApplyPass(st);
        cout << Ok(id, "PASS") << flush;
        continue;
      }

      int best_move = -1;
      int best_score = -1000000000;

      BitPattern rem = moves;
      while (rem) {
        BitPattern sq_bb = rem & (0ULL - rem);
        rem ^= sq_bb;
        int mv = __builtin_ctzll(sq_bb);

        BitPattern flip = GetFlip((Square)mv, st.board.Player(), st.board.Opponent());
        BitPattern next_player = NewPlayer(flip, st.board.Opponent());
        BitPattern next_opp    = NewOpponent(flip, st.board.Player());

        PatternEvaluator pe(evals.data());
        pe.Setup(next_player, next_opp);
        int score = (int)pe.Evaluate();
        score = -score;

        if (score > best_score) { best_score = score; best_move = mv; }
      }

      ApplyMove(st, (Square)best_move);
      cout << Ok(id, SqToCoord((Square)best_move)) << flush;
      continue;
    }
    if (cmd=="final_score") {
      if (!IsOver(st)) { cout << Err(id,"game not over") << flush; continue; }
      auto [nb, nw] = CountDiscsAbs(st);
      int diff = nb - nw;
      string r;
      if (diff > 0) r = "B+" + to_string(diff);
      else if (diff < 0) r = "W+" + to_string(-diff);
      else r = "0";
      cout << Ok(id, r) << flush;
      continue;
    }

    cout << Err(id, "unknown command") << flush;
  }

  return 0;
}
