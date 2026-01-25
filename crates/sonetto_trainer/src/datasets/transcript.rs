use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use sonetto_core::board::{Board, Color, Undo};
use sonetto_core::coord::PASS;
use sonetto_core::eval::{score_disc, Weights};
use sonetto_core::movegen::legal_moves;

use crate::log::RunLog;
use crate::teacher::{TeacherClient, TeacherQuery};
use crate::util::{
    bitpos_to_coord, board_to_compact_65, coord_pair_to_bitpos, parse_board_token, parse_side_token,
    teacher_policy_from_bitpos,
};

/// Aggregate statistics for validation.
#[derive(Debug, Default, Clone)]
pub struct ValStats {
    pub games: u64,
    pub positions: u64,
    pub loss_sum: f64,
    pub skipped_lines: u64,
    pub bad_lines: u64,
    pub incomplete_games: u64,
}

impl ValStats {
    pub fn avg_loss(&self) -> f64 {
        if self.positions == 0 {
            0.0
        } else {
            self.loss_sum / (self.positions as f64)
        }
    }
}

/// Validation mode A: targets are computed from the transcript's final disc difference.
///
/// This is "teacherless" and can run fully offline.
pub fn run_validation_final_score(
    val_dir: &Path,
    weights: &Weights,
    log: &mut RunLog,
    strict: bool,
) -> io::Result<ValStats> {
    let files = list_txt_files(val_dir)?;
    log.line(format!(
        "validation(final_score): scanning {} file(s) under {}",
        files.len(),
        val_dir.display()
    ));

    let mut stats = ValStats::default();

    for path in files {
        let f = File::open(&path)?;
        let br = BufReader::new(f);

        for (lineno, line_res) in br.lines().enumerate() {
            let line = line_res?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                stats.skipped_lines += 1;
                continue;
            }

            match parse_transcript_line(line) {
                Ok(Some((start, moves))) => {
                    match process_game_final_score(start, &moves, weights, &mut stats) {
                        Ok(()) => {}
                        Err(e) => {
                            stats.bad_lines += 1;
                            if strict {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("{}:{}: {e}", path.display(), lineno + 1),
                                ));
                            } else {
                                log.line(format!(
                                    "validation(final_score): skip {}:{}: {e}",
                                    path.display(),
                                    lineno + 1
                                ));
                            }
                        }
                    }
                }
                Ok(None) => {
                    stats.skipped_lines += 1;
                }
                Err(e) => {
                    stats.bad_lines += 1;
                    if strict {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("{}:{}: {e}", path.display(), lineno + 1),
                        ));
                    } else {
                        log.line(format!(
                            "validation(final_score): bad line {}:{}: {e}",
                            path.display(),
                            lineno + 1
                        ));
                    }
                }
            }
        }
    }

    log.line(format!(
        "validation(final_score): done, games={}, positions={}, avg_loss={:.6}",
        stats.games,
        stats.positions,
        stats.avg_loss()
    ));

    Ok(stats)
}

/// Validation mode B: query a teacher server for move evaluations.
///
/// Teacher protocol:
/// - Prefer `POST /get_teacher_guidance` (batched).
/// - Fallback to `POST /eval_moves`.
///
/// This validation uses the *next move* from the transcript as the candidate list.
/// The teacher provides an evaluation for that move; we compare it with Sonetto's
/// egev2 evaluation (converted to the mover's perspective).
pub fn run_validation_teacher(
    val_dir: &Path,
    weights: &Weights,
    teacher: &TeacherClient,
    teacher_batch_size: usize,
    log: &mut RunLog,
    strict: bool,
) -> io::Result<ValStats> {
    let files = list_txt_files(val_dir)?;
    log.line(format!(
        "validation(teacher): scanning {} file(s) under {}",
        files.len(),
        val_dir.display()
    ));

    let mut stats = ValStats::default();

    let mut batch_q: Vec<TeacherQuery> = Vec::with_capacity(teacher_batch_size.max(1));
    let mut batch_pred: Vec<f32> = Vec::with_capacity(teacher_batch_size.max(1));

    for path in files {
        let f = File::open(&path)?;
        let br = BufReader::new(f);

        for (lineno, line_res) in br.lines().enumerate() {
            let line = line_res?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                stats.skipped_lines += 1;
                continue;
            }

            let (mut board, moves) = match parse_transcript_line(line) {
                Ok(Some(v)) => v,
                Ok(None) => {
                    stats.skipped_lines += 1;
                    continue;
                }
                Err(e) => {
                    stats.bad_lines += 1;
                    if strict {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("{}:{}: {e}", path.display(), lineno + 1),
                        ));
                    } else {
                        log.line(format!(
                            "validation(teacher): bad line {}:{}: {e}",
                            path.display(),
                            lineno + 1
                        ));
                        continue;
                    }
                }
            };

            // Process one game.
            stats.games += 1;

            let mut undo = Undo::default();

            for &mv in moves.iter() {
                // implicit pass
                let me = board.player;
                let opp = board.opponent;
                if legal_moves(me, opp) == 0 {
                    board.apply_move(PASS, &mut undo);
                }

                // Build teacher query using the *current* position.
                let board_str = board_to_compact_65(&board);
                let policy = teacher_policy_from_bitpos(mv);
                batch_q.push(TeacherQuery {
                    board: board_str,
                    moves: vec![policy],
                });

                // Apply the move to advance the transcript.
                let ok = board.apply_move(mv, &mut undo);
                if !ok {
                    stats.bad_lines += 1;
                    let msg = format!(
                        "illegal move '{}' (bitpos={mv}) at {}:{}",
                        bitpos_to_coord(mv),
                        path.display(),
                        lineno + 1
                    );
                    if strict {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                    } else {
                        log.line(format!("validation(teacher): skip game: {msg}"));
                        break;
                    }
                }

                // After applying, side toggled. score_disc is from side-to-move perspective,
                // so negate to get the mover's perspective for the just-made move.
                let next_disc = score_disc(&board, weights) as i32;
                let mover_disc = -next_disc;
                let pred = libm::tanhf(mover_disc as f32 / 50.0);
                batch_pred.push(pred);

                // Flush batch if needed.
                if batch_q.len() >= teacher_batch_size.max(1) {
                    flush_teacher_batch(teacher, &mut batch_q, &mut batch_pred, &mut stats, log, strict)?;
                }
            }
        }
    }

    if !batch_q.is_empty() {
        flush_teacher_batch(teacher, &mut batch_q, &mut batch_pred, &mut stats, log, strict)?;
    }

    log.line(format!(
        "validation(teacher): done, games={}, positions={}, avg_loss={:.6}",
        stats.games,
        stats.positions,
        stats.avg_loss()
    ));

    Ok(stats)
}

fn flush_teacher_batch(
    teacher: &TeacherClient,
    batch_q: &mut Vec<TeacherQuery>,
    batch_pred: &mut Vec<f32>,
    stats: &mut ValStats,
    log: &mut RunLog,
    strict: bool,
) -> io::Result<()> {
    let n = batch_q.len();
    if n == 0 {
        return Ok(());
    }

    debug_assert_eq!(n, batch_pred.len());

    let res = teacher.eval_moves_batch(batch_q);
    let evals = match res {
        Ok(v) => v,
        Err(e) => {
            if strict {
                return Err(io::Error::new(io::ErrorKind::Other, e.to_string()));
            } else {
                log.line(format!("validation(teacher): teacher error (skip {n} positions): {e}"));
                batch_q.clear();
                batch_pred.clear();
                return Ok(());
            }
        }
    };

    for i in 0..n {
        let teacher_disc = evals
            .get(i)
            .and_then(|x| x.get(0))
            .copied()
            .unwrap_or(0);

        let teacher_t = libm::tanhf(teacher_disc as f32 / 50.0);
        let d = batch_pred[i] - teacher_t;

        stats.positions += 1;
        stats.loss_sum += (d * d) as f64;
    }

    batch_q.clear();
    batch_pred.clear();
    Ok(())
}

fn process_game_final_score(
    mut board: Board,
    moves: &[u8],
    weights: &Weights,
    stats: &mut ValStats,
) -> Result<(), String> {
    let mut preds: Vec<(Color, f32)> = Vec::with_capacity(moves.len());
    let mut undo = Undo::default();

    for &mv in moves.iter() {
        // implicit pass if no legal moves
        let me = board.player;
        let opp = board.opponent;
        if legal_moves(me, opp) == 0 {
            board.apply_move(PASS, &mut undo);
        }

        // prediction for current position (side-to-move perspective)
        let pred_disc = score_disc(&board, weights) as f32;
        let y = libm::tanhf(pred_disc / 50.0);
        preds.push((board.side, y));

        // apply move
        let ok = board.apply_move(mv, &mut undo);
        if !ok {
            return Err(format!("illegal move '{}' (bitpos={mv})", bitpos_to_coord(mv)));
        }
    }

    // Require that transcript reaches terminal: no moves for both players.
    let me = board.player;
    let opp = board.opponent;
    if legal_moves(me, opp) != 0 {
        stats.incomplete_games += 1;
        return Ok(());
    }

    board.apply_move(PASS, &mut undo);

    let me2 = board.player;
    let opp2 = board.opponent;
    if legal_moves(me2, opp2) != 0 {
        stats.incomplete_games += 1;
        return Ok(());
    }

    // Final disc difference from black perspective.
    let bc = board.bits_of(Color::Black).count_ones() as i32;
    let wc = board.bits_of(Color::White).count_ones() as i32;
    let final_diff_black = bc - wc;

    // Convert to per-position targets (side-to-move perspective).
    for (side, y) in preds.iter() {
        let target = if *side == Color::Black {
            final_diff_black
        } else {
            -final_diff_black
        };
        let t = libm::tanhf(target as f32 / 50.0);
        let d = *y - t;

        stats.positions += 1;
        stats.loss_sum += (d * d) as f64;
    }

    stats.games += 1;
    Ok(())
}

fn parse_transcript_line(line: &str) -> Result<Option<(Board, Vec<u8>)>, String> {
    // Remove leading/trailing whitespace and split.
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(None);
    }

    // Case A: "board64 side moves..."
    // Case B: "board65 moves..." (board+side compact)
    // Case C: "moves..." (implicit startpos, Black)
    let p0 = parts[0];

    if p0.len() == 64 {
        if parts.len() < 3 {
            return Err("expected: <board64> <side> <moves>".to_string());
        }
        let side = parse_side_token(parts[1])
            .ok_or_else(|| format!("invalid side token: '{}'", parts[1]))?;
        let start = parse_board_token(p0, side)?;
        let moves_str = parts[2..].join("");
        let moves = parse_moves_concat(&moves_str)?;
        return Ok(Some((start, moves)));
    }

    if p0.len() == 65 {
        if parts.len() < 2 {
            return Err("expected: <board65> <moves>".to_string());
        }
        let start = parse_board_token(p0, Color::Black)?;
        let moves_str = parts[1..].join("");
        let moves = parse_moves_concat(&moves_str)?;
        return Ok(Some((start, moves)));
    }

    // Default: treat the line as moves only, starting from initial board.
    let mut start = Board::new_start(0);
    // Moves only implies black to move.
    start.side = Color::Black;
    let moves_str = parts.join("");
    let moves = parse_moves_concat(&moves_str)?;
    Ok(Some((start, moves)))
}

fn parse_moves_concat(moves: &str) -> Result<Vec<u8>, String> {
    // Remove any whitespace inside (defensive).
    let s: String = moves.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = s.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(format!("moves string length must be even, got {}", bytes.len()));
    }

    let mut out: Vec<u8> = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let file = bytes[i];
        let rank = bytes[i + 1];

        let mv = coord_pair_to_bitpos(file, rank)
            .ok_or_else(|| format!("invalid move pair: '{}{}'", file as char, rank as char))?;
        out.push(mv);
        i += 2;
    }
    Ok(out)
}

fn list_txt_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();

    let rd = match fs::read_dir(dir) {
        Ok(x) => x,
        Err(e) => {
            return Err(io::Error::new(
                e.kind(),
                format!("unable to read validation dir {}: {e}", dir.display()),
            ))
        }
    };

    for ent in rd {
        let ent = ent?;
        let p = ent.path();
        if p.is_file() {
            if let Some(ext) = p.extension() {
                if ext == "txt" {
                    out.push(p);
                }
            }
        }
    }

    out.sort();
    Ok(out)
}
