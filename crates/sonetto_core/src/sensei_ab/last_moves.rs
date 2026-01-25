//! Exact endgame evaluation for a very small number of empty squares.
//!
//! Native Sensei contains a specialized, fully unrolled solver for the last
//! 1..5 empties. This module ports that idea to Sonetto's bitboard mapping.
//!
//! Differences vs generic search:
//! - avoids move list allocation / iterator overhead
//! - uses hard-unrolled alpha-beta over the remaining empties
//! - includes Sensei-style heuristics:
//!   - stability upper-bound pruning (opponent stable disks)
//!   - 5-empties quadrant-based empty ordering
//!
//! Returned values are **disc-difference** in [-64, 64] from the current
//! side-to-move perspective (same unit as `score_disc`).

use crate::coord::Move;
use crate::flips::flips_for_move;
use crate::stability::stable_disks_definitely;

/// Sensei-style sentinel: "no move found".
pub(crate) const LESS_THAN_MIN_EVAL: i32 = -66;

// --- Patterns for Sensei's 5-empties ordering (Sonetto bit mapping: A1 = bit 0). ---

const CORNER_PATTERN: u64 = 0x8100_0000_0000_0081;
const CENTRAL_PATTERN: u64 = 0x003c_7e7e_7e7e_3c00;
const EDGE_PATTERN: u64 = 0x3c00_8181_8181_003c;
// "XC" squares (X + C) around corners, but excluding the corners themselves.
const XC_PATTERN: u64 = 0x42c3_0000_0000_c342;

// Quadrants (4x4 spaces) used by Sensei's 5-empties ordering.
const SPACE0_PATTERN: u64 = 0x0000_0000_f0f0_f0f0; // bottom-right (E1..H4)
const SPACE1_PATTERN: u64 = 0x0000_0000_0f0f_0f0f; // bottom-left  (A1..D4)
const SPACE2_PATTERN: u64 = 0xf0f0_f0f0_0000_0000; // top-right    (E5..H8)
const SPACE3_PATTERN: u64 = 0x0f0f_0f0f_0000_0000; // top-left     (A5..D8)

#[inline(always)]
fn game_over_disc_from_bits(me: u64, opp: u64) -> i32 {
    let me_discs = me.count_ones() as i32;
    let opp_discs = opp.count_ones() as i32;
    let diff = me_discs - opp_discs;
    let empties = 64 - me_discs - opp_discs;
    if diff > 0 {
        diff + empties
    } else if diff < 0 {
        diff - empties
    } else {
        0
    }
}

#[inline(always)]
fn stability_upper_bound_disc(player: u64, opponent: u64) -> i32 {
    // Upper bound from current player's perspective based on definitely-stable opponent discs.
    let stable_opp = stable_disks_definitely(opponent, player);
    64 - 2 * (stable_opp.count_ones() as i32)
}

#[inline(always)]
fn new_player_opponent_after_flip(player: u64, opponent: u64, flip_incl_move: u64) -> (u64, u64) {
    // Sensei convention: `flip_incl_move` includes the move bit itself.
    // After a move, side-to-move becomes the old opponent.
    let new_player = opponent & !flip_incl_move;
    let new_opponent = player | flip_incl_move;
    (new_player, new_opponent)
}

#[inline(always)]
fn flip_incl_move_for(player: u64, opponent: u64, mv: Move) -> u64 {
    let mv_bit = 1u64 << mv;
    let flips = flips_for_move(player, opponent, mv_bit);
    if flips == 0 {
        0
    } else {
        flips | mv_bit
    }
}

/// Exact evaluation with exactly **one** empty square.
#[inline(always)]
pub(crate) fn eval_one_empty(x: Move, player: u64, opponent: u64) -> i32 {
    let flip = flip_incl_move_for(player, opponent, x);
    if flip != 0 {
        // Current player can play.
        let (_new_player, new_opponent) = new_player_opponent_after_flip(player, opponent, flip);
        return (new_opponent.count_ones() as i32) * 2 - 64;
    }

    // Pass: try opponent.
    let flip = flip_incl_move_for(opponent, player, x);
    if flip != 0 {
        // Opponent plays; compute from *current* player's perspective.
        let (_new_player, new_opponent) = new_player_opponent_after_flip(opponent, player, flip);
        // `new_opponent` here are the discs of the (opponent) mover.
        let opp_discs = new_opponent.count_ones() as i32;
        return 64 - 2 * opp_discs;
    }

    // No legal move for either side.
    game_over_disc_from_bits(player, opponent)
}

#[inline(always)]
fn eval_two_empties_or_min(
    x1: Move,
    x2: Move,
    player: u64,
    opponent: u64,
    lower: i32,
    upper: i32,
    visited: &mut u64,
) -> i32 {
    *visited += 1;
    let mut eval = LESS_THAN_MIN_EVAL;

    let flip = flip_incl_move_for(player, opponent, x1);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        eval = -eval_one_empty(x2, np, no);
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x2);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = -eval_one_empty(x1, np, no);
        if v > eval {
            eval = v;
        }
        return eval;
    }

    eval
}

#[inline(always)]
pub(crate) fn eval_two_empties(
    x1: Move,
    x2: Move,
    player: u64,
    opponent: u64,
    lower: i32,
    upper: i32,
    visited: &mut u64,
) -> i32 {
    let eval = eval_two_empties_or_min(x1, x2, player, opponent, lower, upper, visited);
    if eval > LESS_THAN_MIN_EVAL {
        return eval;
    }
    let eval = eval_two_empties_or_min(x1, x2, opponent, player, -upper, -lower, visited);
    if eval > LESS_THAN_MIN_EVAL {
        return -eval;
    }
    game_over_disc_from_bits(player, opponent)
}

#[inline(always)]
fn eval_three_empties_or_min(
    x1: Move,
    x2: Move,
    x3: Move,
    player: u64,
    opponent: u64,
    lower: i32,
    upper: i32,
    visited: &mut u64,
) -> i32 {
    *visited += 1;
    let mut eval = LESS_THAN_MIN_EVAL;

    let flip = flip_incl_move_for(player, opponent, x1);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        eval = -eval_two_empties(x2, x3, np, no, -upper, -lower, visited);
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x2);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = -eval_two_empties(x1, x3, np, no, -upper, -lower.max(eval), visited);
        if v > eval {
            eval = v;
        }
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x3);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = -eval_two_empties(x1, x2, np, no, -upper, -lower.max(eval), visited);
        if v > eval {
            eval = v;
        }
        return eval;
    }

    eval
}

#[inline(always)]
pub(crate) fn eval_three_empties(
    x1: Move,
    x2: Move,
    x3: Move,
    player: u64,
    opponent: u64,
    lower: i32,
    upper: i32,
    visited: &mut u64,
) -> i32 {
    let eval = eval_three_empties_or_min(x1, x2, x3, player, opponent, lower, upper, visited);
    if eval > LESS_THAN_MIN_EVAL {
        return eval;
    }
    let eval = eval_three_empties_or_min(x1, x2, x3, opponent, player, -upper, -lower, visited);
    if eval > LESS_THAN_MIN_EVAL {
        return -eval;
    }
    game_over_disc_from_bits(player, opponent)
}

#[inline(always)]
fn eval_four_empties_or_min(
    x1: Move,
    x2: Move,
    x3: Move,
    x4: Move,
    player: u64,
    opponent: u64,
    lower: i32,
    upper: i32,
    swap: bool,
    visited: &mut u64,
) -> i32 {
    *visited += 1;
    let mut eval = LESS_THAN_MIN_EVAL;

    let flip = flip_incl_move_for(player, opponent, x1);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        eval = -eval_three_empties(x2, x3, x4, np, no, -upper, -lower, visited);
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x2);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = -eval_three_empties(x1, x3, x4, np, no, -upper, -lower.max(eval), visited);
        if v > eval {
            eval = v;
        }
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x3);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = if swap {
            -eval_three_empties(x4, x1, x2, np, no, -upper, -lower.max(eval), visited)
        } else {
            -eval_three_empties(x1, x2, x4, np, no, -upper, -lower.max(eval), visited)
        };
        if v > eval {
            eval = v;
        }
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x4);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = if swap {
            -eval_three_empties(x3, x1, x2, np, no, -upper, -lower.max(eval), visited)
        } else {
            -eval_three_empties(x1, x2, x3, np, no, -upper, -lower.max(eval), visited)
        };
        if v > eval {
            eval = v;
        }
        return eval;
    }

    eval
}

/// Exact evaluation with exactly **four** empty squares.
///
/// `swap` is the Sensei heuristic flag produced by 5-empties ordering.
/// `last_flip_incl_move` is the move that led into this node (Sensei flip convention).
#[inline(always)]
pub(crate) fn eval_four_empties(
    x1: Move,
    x2: Move,
    x3: Move,
    x4: Move,
    player: u64,
    opponent: u64,
    lower: i32,
    upper: i32,
    swap: bool,
    _last_flip_incl_move: u64,
    visited: &mut u64,
) -> i32 {
    // Stability upper-bound cutoff (Sensei). This is safe pruning: if the best
    // achievable outcome is already <= lower, we can return the bound.
    let stability_cutoff_upper = stability_upper_bound_disc(player, opponent);
    if stability_cutoff_upper <= lower {
        return stability_cutoff_upper;
    }

    let eval = eval_four_empties_or_min(x1, x2, x3, x4, player, opponent, lower, upper, swap, visited);
    if eval > LESS_THAN_MIN_EVAL {
        return eval;
    }
    let eval = eval_four_empties_or_min(x1, x2, x3, x4, opponent, player, -upper, -lower, swap, visited);
    if eval > LESS_THAN_MIN_EVAL {
        return -eval;
    }
    game_over_disc_from_bits(player, opponent)
}

#[inline(always)]
fn eval_five_empties_or_min(
    x1: Move,
    x2: Move,
    x3: Move,
    x4: Move,
    x5: Move,
    player: u64,
    opponent: u64,
    lower: i32,
    upper: i32,
    swap: bool,
    visited: &mut u64,
) -> i32 {
    *visited += 1;
    let mut eval = LESS_THAN_MIN_EVAL;

    let flip = flip_incl_move_for(player, opponent, x1);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        eval = -eval_four_empties(x2, x3, x4, x5, np, no, -upper, -lower, swap, flip, visited);
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x2);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = -eval_four_empties(x1, x3, x4, x5, np, no, -upper, -lower.max(eval), swap, flip, visited);
        if v > eval {
            eval = v;
        }
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x3);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = -eval_four_empties(x1, x2, x4, x5, np, no, -upper, -lower.max(eval), swap, flip, visited);
        if v > eval {
            eval = v;
        }
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x4);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = if swap {
            -eval_four_empties(x5, x2, x3, x1, np, no, -upper, -lower.max(eval), swap, flip, visited)
        } else {
            -eval_four_empties(x1, x2, x3, x5, np, no, -upper, -lower.max(eval), swap, flip, visited)
        };
        if v > eval {
            eval = v;
        }
        if eval >= upper {
            return eval;
        }
    }

    let flip = flip_incl_move_for(player, opponent, x5);
    if flip != 0 {
        let (np, no) = new_player_opponent_after_flip(player, opponent, flip);
        let v = if swap {
            -eval_four_empties(x4, x2, x3, x1, np, no, -upper, -lower.max(eval), swap, flip, visited)
        } else {
            -eval_four_empties(x1, x2, x3, x4, np, no, -upper, -lower.max(eval), swap, flip, visited)
        };
        if v > eval {
            eval = v;
        }
        return eval;
    }

    eval
}

#[inline(always)]
fn reorder_five_empties(empties: u64) -> ([Move; 5], bool) {
    debug_assert_eq!(empties.count_ones(), 5);

    let mut x: [Move; 5] = [0; 5];
    let mut empties_left = empties;

    let mut cont_x: usize = 0;
    let mut new_cont_x: usize = 0;

    let mut has_space_2 = false;
    let mut has_space_3 = false;
    let mut space_3: [Move; 3] = [0; 3];

    const SPACES: [u64; 4] = [SPACE0_PATTERN, SPACE1_PATTERN, SPACE2_PATTERN, SPACE3_PATTERN];
    const MASKS: [u64; 4] = [CORNER_PATTERN, CENTRAL_PATTERN, EDGE_PATTERN, XC_PATTERN];

    for &space in &SPACES {
        let empties_in_space = space & empties_left;
        if empties_in_space == 0 {
            continue;
        }

        let space_size = empties_in_space.count_ones();
        if space_size == 1 {
            let sq = empties_in_space.trailing_zeros() as u8;
            x[cont_x] = sq;
            cont_x += 1;
            empties_left &= !empties_in_space;
            continue;
        } else if space_size == 2 {
            new_cont_x = cont_x;
            cont_x = if has_space_2 { 1 } else { 3 };
            has_space_2 = true;
        } else if space_size == 3 {
            has_space_3 = true;
            let mut j = 0usize;
            for &mask in &MASKS {
                let mut m = empties_in_space & mask;
                while m != 0 {
                    let sq = m.trailing_zeros() as u8;
                    space_3[j] = sq;
                    j += 1;
                    m &= m - 1;
                }
            }
            empties_left &= !empties_in_space;
            continue;
        } else if space_size == 4 {
            new_cont_x = 0;
            cont_x = 1;
        }

        for &mask in &MASKS {
            let mut m = empties_in_space & mask;
            while m != 0 {
                let sq = m.trailing_zeros() as u8;
                x[cont_x] = sq;
                cont_x += 1;
                m &= m - 1;
            }
        }

        empties_left &= !empties_in_space;
        cont_x = new_cont_x;
    }

    // The 3-in-space case is placed either at the front or at the end, depending on whether
    // we also saw a 2-in-space (Sensei's `has_space_2` / `swap`).
    if has_space_3 {
        if has_space_2 {
            x[0] = space_3[0];
            x[1] = space_3[1];
            x[2] = space_3[2];
        } else {
            x[2] = space_3[0];
            x[3] = space_3[1];
            x[4] = space_3[2];
        }
    }

    (x, has_space_2)
}

/// Exact evaluation with exactly **five** empty squares.
///
/// This is the entry point used by the Sensei AB endgame solver.
#[inline(always)]
pub(crate) fn eval_five_empties(
    player: u64,
    opponent: u64,
    lower: i32,
    upper: i32,
    _last_flip_incl_move: u64,
    visited: &mut u64,
) -> i32 {
    // Stability upper-bound cutoff (Sensei).
    let stability_cutoff_upper = stability_upper_bound_disc(player, opponent);
    if stability_cutoff_upper <= lower {
        return stability_cutoff_upper;
    }

    let empties = !(player | opponent);
    let (x, swap) = reorder_five_empties(empties);

    let eval = eval_five_empties_or_min(x[0], x[1], x[2], x[3], x[4], player, opponent, lower, upper, swap, visited);
    if eval > LESS_THAN_MIN_EVAL {
        return eval;
    }

    // Pass: try opponent.
    let eval = eval_five_empties_or_min(x[0], x[1], x[2], x[3], x[4], opponent, player, -upper, -lower, swap, visited);
    if eval > LESS_THAN_MIN_EVAL {
        return -eval;
    }

    game_over_disc_from_bits(player, opponent)
}
