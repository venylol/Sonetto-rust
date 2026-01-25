//! Alpha-beta search (Sonetto-style negamaxHybrid core):
//! - Negamax
//! - PVS (principal variation search)
//! - IID: d>=4 && !hasTT && mc>1  (shallow search to pick a PV move)
//! - LMR: d>=3 && i>=3 && empty>24  => reduce by 1 ply
//! - Killer moves + History heuristic ordering
//! - NO heap allocation inside recursive search (uses preallocated arrays)

use crate::board::{Board, Undo};
use crate::coord::{Move, PASS};
use crate::eval::{build_sonetto_feature_defs_and_occ, evaluate, FeatureDefs, Weights};
use crate::features::occ::OccMap;
use crate::features::swap::SwapTables;
use crate::features::update::recompute_features_in_place;
use crate::flips::{flips_for_move, flips_for_move_unchecked};
use crate::movegen::{legal_moves, NOT_A, NOT_H};
use crate::score::{disc_diff_scaled, game_over_scaled, game_over_scaled_from_bits, eval1000_to_score, Score, SCALE};
use crate::tt::{Bound, TranspositionTable};

// P2-4: Rayon-based parallel helpers (used for WASM internal threading via
// wasm-bindgen-rayon, and optionally for native builds).
#[cfg(feature = "parallel_rayon")]
use rayon::prelude::*;

#[cfg(feature = "stability_cutoff")]
use crate::stability::stability_bounds_for_side_to_move;

pub const MAX_PLY: usize = 64;
/// Maximum number of moves we can hold in the per-ply move buffers.
///
/// NOTE: Some (especially adversarial/editor-created) Othello positions can
/// have a surprisingly large number of legal moves.
///
/// We therefore provision the *absolute* board upper bound (64) to avoid any
/// truncation.
pub const MAX_MOVES: usize = 64;

const INF: Score = 1_000_000;

// -----------------------------------------------------------------------------
// P1-2: Multi-ProbCut (Egaroucid-inspired)
// -----------------------------------------------------------------------------
//
// A conservative, engine-friendly implementation:
// - Only applied away from the root (ply >= 5) and at sufficiently large depths.
// - Uses a cheap disc-diff prefilter before paying for a full eval().
// - Uses a single shallow null-window probe search.
// - Nested ProbCut probes are disabled via `probcut_nesting`.
//
// References (for the regression constants and overall shape):
// - Egaroucid `probcut.hpp` (GPL-3.0-or-later)

const PROBCUT_SHALLOW_IGNORE: usize = 5;
const PROBCUT_MIN_DEPTH: u8 = 6;
const PROBCUT_REDUCTION_SHALLOW: u8 = 2;
const PROBCUT_REDUCTION_DEEP: u8 = 3;

const PROBCUT_MPCT: f64 = 1.18;

// Disc-diff prefilter: require the *raw* disc difference to already be far
// beyond alpha/beta before we compute a full pattern eval for ProbCut.
const PROBCUT_QUICK_TRIGGER: Score = 6 * SCALE; // ~6 discs

// Regression constants copied from Egaroucid's `probcut.hpp`.
const PROBCUT_A: f64 = 0.3921669389943707;
const PROBCUT_B: f64 = -1.9069468919346821;
const PROBCUT_C: f64 = 1.9789690637551312;
const PROBCUT_D: f64 = 1.3837874301234074;
const PROBCUT_E: f64 = -5.5248567821753705;
const PROBCUT_F: f64 = 13.018474287421077;
const PROBCUT_G: f64 = 8.5736852851878003;

#[inline(always)]
fn probcut_sigma(n_discs: i32, depth1: i32, depth2: i32) -> f64 {
    let mut x = PROBCUT_A * (n_discs as f64 / 64.0)
        + PROBCUT_B * (depth1 as f64 / 60.0)
        + PROBCUT_C * (depth2 as f64 / 60.0);

    // cubic polynomial fit
    x = PROBCUT_D * x * x * x + PROBCUT_E * x * x + PROBCUT_F * x + PROBCUT_G;
    x
}

#[inline(always)]
fn probcut_sigma_depth0(n_discs: i32, depth2: i32) -> f64 {
    let mut x = PROBCUT_A * (n_discs as f64 / 64.0)
        + PROBCUT_C * (depth2 as f64 / 60.0);

    x = PROBCUT_D * x * x * x + PROBCUT_E * x * x + PROBCUT_F * x + PROBCUT_G;
    x
}

#[inline(always)]
fn probcut_error_disc_depth0(n_discs: i32, depth: u8) -> i32 {
    let sigma = probcut_sigma_depth0(n_discs, depth as i32);
    let e = (PROBCUT_MPCT * sigma).ceil() as i32;
    if e <= 0 {
        0
    } else if e >= 64 {
        64
    } else {
        e
    }
}

#[inline(always)]
fn probcut_error_disc_search(n_discs: i32, depth: u8, search_depth: u8) -> i32 {
    let sigma = probcut_sigma(n_discs, depth as i32, search_depth as i32);
    let e = (PROBCUT_MPCT * sigma).ceil() as i32;
    if e <= 0 {
        0
    } else if e >= 64 {
        64
    } else {
        e
    }
}

#[inline(always)]
fn probcut_error_score_depth0(n_discs: i32, depth: u8) -> Score {
    probcut_error_disc_depth0(n_discs, depth) * SCALE
}

#[inline(always)]
fn probcut_error_score_search(n_discs: i32, depth: u8, search_depth: u8) -> Score {
    probcut_error_disc_search(n_discs, depth, search_depth) * SCALE
}


// -----------------------------------------------------------------------------
// Stage 4: iterative analysis + node budget (Derivative-style Progress/Advancement)
// -----------------------------------------------------------------------------

/// Absolute safe score bound in Sonetto's `Score` units.
///
/// All evaluation / terminal scores are clamped into `[-SCORE_BOUND, SCORE_BOUND]`.
const SCORE_BOUND: Score = 64 * SCALE; // 2048

const SCORE_MIN: Score = -SCORE_BOUND;
const SCORE_MAX: Score = SCORE_BOUND;


// -----------------------------------------------------------------------------
// P1-1: Endgame helpers (NWS/PVS exact solver + parity ordering + lastN + local TT)
// -----------------------------------------------------------------------------

/// Local TT size used by the endgame solver.
///
/// This matches Egaroucid's default (1024) and is intended to fit in L1 cache.
const ENDGAME_LTT_SIZE: usize = 1024;
const ENDGAME_LTT_BITS: u32 = 10; // log2(ENDGAME_LTT_SIZE)

/// Only use the local endgame TT when empties are small (typical exact solve range).
const ENDGAME_LTT_MAX_EMPTIES: u8 = 20;

/// When empties are small, apply a parity-based move ordering (4x4 quadrant parity).
const ENDGAME_PARITY_ORDER_MAX_EMPTIES: u8 = 20;

/// 4x4 quadrant masks (row-major bitpos).
const PARITY_Q0_MASK: u64 = 0x0000_0000_0F0F_0F0F;
const PARITY_Q1_MASK: u64 = 0x0000_0000_F0F0_F0F0;
const PARITY_Q2_MASK: u64 = 0x0F0F_0F0F_0000_0000;
const PARITY_Q3_MASK: u64 = 0xF0F0_F0F0_0000_0000;

/// For each square, which 4x4 quadrant it belongs to (bitmask {1,2,4,8}).
///
/// Copied from Egaroucid's `cell_div4` table.
const CELL_DIV4: [u8; 64] = [
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
];

#[inline(always)]
fn parity_bits_from_empties(empties: u64) -> u8 {
    let mut p: u8 = 0;
    if ((empties & PARITY_Q0_MASK).count_ones() & 1) != 0 {
        p |= 1;
    }
    if ((empties & PARITY_Q1_MASK).count_ones() & 1) != 0 {
        p |= 2;
    }
    if ((empties & PARITY_Q2_MASK).count_ones() & 1) != 0 {
        p |= 4;
    }
    if ((empties & PARITY_Q3_MASK).count_ones() & 1) != 0 {
        p |= 8;
    }
    p
}

#[inline(always)]
fn end_ltt_hash_index(me: u64, opp: u64) -> usize {
    // Same mixing constants as Egaroucid (good avalanche, cheap).
    const C1: u64 = 0x9dda_1c54_cfe6_b6e9;
    const C2: u64 = 0xa2e6_c030_0831_e05a;

    let h = me.wrapping_mul(C1) ^ opp.wrapping_mul(C2);
    let idx = (h >> (64 - ENDGAME_LTT_BITS)) as usize;
    idx & (ENDGAME_LTT_SIZE - 1)
}

#[derive(Clone, Copy)]
struct EndLocalTTEntry {
    me: u64,
    opp: u64,
    lower: Score,
    upper: Score,
    best_move: Move,
}

impl Default for EndLocalTTEntry {
    fn default() -> Self {
        Self {
            me: 0,
            opp: 0,
            lower: SCORE_MIN,
            upper: SCORE_MAX,
            best_move: PASS,
        }
    }
}

/// Tiny per-searcher local TT for endgame solves.
///
/// This is deliberately *very* small and hot in cache. It stores a pair of
/// bounds (lower/upper) and a best move, keyed by `(me, opp)` bitboards.
struct EndLocalTT {
    table: Vec<EndLocalTTEntry>,
}

impl EndLocalTT {
    fn new() -> Self {
        let depth_count = (ENDGAME_LTT_MAX_EMPTIES as usize) + 1;
        Self {
            table: vec![EndLocalTTEntry::default(); depth_count * ENDGAME_LTT_SIZE],
        }
    }

    #[inline(always)]
    fn idx(empties: u8, me: u64, opp: u64) -> Option<usize> {
        if empties > ENDGAME_LTT_MAX_EMPTIES {
            return None;
        }
        let h = end_ltt_hash_index(me, opp);
        Some((empties as usize) * ENDGAME_LTT_SIZE + h)
    }

    #[inline(always)]
    fn probe(&self, empties: u8, me: u64, opp: u64) -> Option<&EndLocalTTEntry> {
        let idx = Self::idx(empties, me, opp)?;
        let e = &self.table[idx];
        if e.me == me && e.opp == opp {
            Some(e)
        } else {
            None
        }
    }

    #[inline(always)]
    fn store(&mut self, empties: u8, me: u64, opp: u64, flag: Bound, value: Score, best_move: Move) {
        let Some(idx) = Self::idx(empties, me, opp) else {
            return;
        };

        let e = &mut self.table[idx];

        // Replace on mismatch.
        if e.me != me || e.opp != opp {
            *e = EndLocalTTEntry {
                me,
                opp,
                lower: SCORE_MIN,
                upper: SCORE_MAX,
                best_move,
            };
        }

        match flag {
            Bound::Exact => {
                e.lower = value;
                e.upper = value;
                e.best_move = best_move;
            }
            Bound::Lower => {
                if value > e.lower {
                    e.lower = value;
                    e.best_move = best_move;
                }
            }
            Bound::Upper => {
                if value < e.upper {
                    e.upper = value;
                    // Keep best_move as-is; upper bounds usually come from fail-low.
                }
            }
        }
    }
}


/// Solve the last 1 empty square exactly (no recursion, no TT).
///
/// Inputs are from the *current side-to-move* perspective:
/// - `me`: discs of side to move
/// - `opp`: discs of opponent
/// - `empty_bit`: the only empty square (single bit)
#[inline(always)]
fn exact_last1(me: u64, opp: u64, empty_bit: u64) -> (Score, Move) {
    debug_assert!(empty_bit != 0);
    debug_assert!(empty_bit & (empty_bit - 1) == 0);

    // Can we play on the last empty?
    let flips = flips_for_move(me, opp, empty_bit);
    if flips != 0 {
        let new_me = me | empty_bit | flips;
        let new_opp = opp & !flips;
        let mv = empty_bit.trailing_zeros() as Move;
        return (game_over_scaled_from_bits(new_me, new_opp), mv);
    }

    // PASS: can the opponent play?
    let flips2 = flips_for_move(opp, me, empty_bit);
    if flips2 != 0 {
        let new_opp_me = opp | empty_bit | flips2;
        let new_opp_opp = me & !flips2;
        return (game_over_scaled_from_bits(new_opp_opp, new_opp_me), PASS);
    }

    // Game over with one empty left.
    (game_over_scaled_from_bits(me, opp), PASS)
}

/// Solve the last 2 empty squares exactly.
///
/// This avoids generating full legal move masks and only tests the 2 empty squares.
#[inline(always)]
fn exact_last2(me: u64, opp: u64, empties: u64) -> (Score, Move) {
    debug_assert!(empties.count_ones() == 2);

    // Extract the least-significant set bit.
    //
    // Use `wrapping_neg` to avoid any potential overflow panics in debug builds
    // (e.g. if this helper is ever reused with `empties == 0`).
    let p0_bit = empties & empties.wrapping_neg();
    let p1_bit = empties ^ p0_bit;

    let flips0 = flips_for_move(me, opp, p0_bit);
    let flips1 = flips_for_move(me, opp, p1_bit);

    // No moves -> PASS or terminal
    if flips0 == 0 && flips1 == 0 {
        let opp_flips0 = flips_for_move(opp, me, p0_bit);
        let opp_flips1 = flips_for_move(opp, me, p1_bit);

        if opp_flips0 == 0 && opp_flips1 == 0 {
            return (game_over_scaled_from_bits(me, opp), PASS);
        }

        // PASS: opponent plays, then we solve last1.
        // Equivalent to: score = -exact_last2(opp, me, empties).0
        let (opp_score, _opp_mv) = exact_last2(opp, me, empties);
        return (-opp_score, PASS);
    }

    // Try legal moves (max stage).
    let mut best: Score = SCORE_MIN;
    let mut best_move: Move = PASS;

    if flips0 != 0 {
        let new_me = me | p0_bit | flips0;
        let new_opp = opp & !flips0;
        let (child, _mv) = exact_last1(new_opp, new_me, p1_bit);
        let score = -child;
        best = score;
        best_move = p0_bit.trailing_zeros() as Move;
    }

    if flips1 != 0 {
        let new_me = me | p1_bit | flips1;
        let new_opp = opp & !flips1;
        let (child, _mv) = exact_last1(new_opp, new_me, p0_bit);
        let score = -child;
        if best_move == PASS || score > best {
            best = score;
            best_move = p1_bit.trailing_zeros() as Move;
        }
    }

    (best, best_move)
}


// ---------------------------------------------------------------------------
// Last N (3/4/5) exact endgame fast paths (P1-5)
// ---------------------------------------------------------------------------

/// Exact solve with last 3 empties.
///
/// This is a small recursive search on *bitboards only* (no Board make/undo, no TT),
/// intended to remove overhead in the very last plies.
#[inline(always)]
fn exact_last3(me: u64, opp: u64, empties: u64) -> (Score, Move) {
    debug_assert!(empties.count_ones() == 3);

    // Extract the three empty squares as single-bit masks.
    let p0_bit = empties & empties.wrapping_neg();
    let e1 = empties ^ p0_bit;
    let p1_bit = e1 & e1.wrapping_neg();
    let p2_bit = e1 ^ p1_bit;

    // Compute flips (0 => illegal move).
    let flips0 = flips_for_move_unchecked(me, opp, p0_bit);
    let flips1 = flips_for_move_unchecked(me, opp, p1_bit);
    let flips2 = flips_for_move_unchecked(me, opp, p2_bit);

    // No moves: PASS or terminal.
    if flips0 == 0 && flips1 == 0 && flips2 == 0 {
        let opp_flips0 = flips_for_move_unchecked(opp, me, p0_bit);
        let opp_flips1 = flips_for_move_unchecked(opp, me, p1_bit);
        let opp_flips2 = flips_for_move_unchecked(opp, me, p2_bit);

        // Both sides have no moves => game over (empties remain empty).
        if opp_flips0 == 0 && opp_flips1 == 0 && opp_flips2 == 0 {
            return (game_over_scaled_from_bits(me, opp), PASS);
        }

        // PASS: same empties, swap sides.
        let (opp_score, _mv) = exact_last3(opp, me, empties);
        return (-opp_score, PASS);
    }

    let mut best: Score = SCORE_MIN;
    let mut best_move: Move = PASS;

    // Try each legal move; recurse into last2.
    if flips0 != 0 {
        let new_me = me | p0_bit | flips0;
        let new_opp = opp & !flips0;
        let (child, _mv) = exact_last2(new_opp, new_me, empties ^ p0_bit);
        let score = -child;

        best = score;
        best_move = p0_bit.trailing_zeros() as Move;
    }
    if flips1 != 0 {
        let new_me = me | p1_bit | flips1;
        let new_opp = opp & !flips1;
        let (child, _mv) = exact_last2(new_opp, new_me, empties ^ p1_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p1_bit.trailing_zeros() as Move;
        }
    }
    if flips2 != 0 {
        let new_me = me | p2_bit | flips2;
        let new_opp = opp & !flips2;
        let (child, _mv) = exact_last2(new_opp, new_me, empties ^ p2_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p2_bit.trailing_zeros() as Move;
        }
    }

    (best, best_move)
}

/// Exact solve with last 4 empties.
///
/// This is the same idea as [`exact_last3`], recursing into it.
#[inline(always)]
fn exact_last4(me: u64, opp: u64, empties: u64) -> (Score, Move) {
    debug_assert!(empties.count_ones() == 4);

    let mut e = empties;

    let p0_bit = e & e.wrapping_neg();
    e ^= p0_bit;
    let p1_bit = e & e.wrapping_neg();
    e ^= p1_bit;
    let p2_bit = e & e.wrapping_neg();
    e ^= p2_bit;
    let p3_bit = e; // last remaining bit

    let flips0 = flips_for_move_unchecked(me, opp, p0_bit);
    let flips1 = flips_for_move_unchecked(me, opp, p1_bit);
    let flips2 = flips_for_move_unchecked(me, opp, p2_bit);
    let flips3 = flips_for_move_unchecked(me, opp, p3_bit);

    if flips0 == 0 && flips1 == 0 && flips2 == 0 && flips3 == 0 {
        let opp_flips0 = flips_for_move_unchecked(opp, me, p0_bit);
        let opp_flips1 = flips_for_move_unchecked(opp, me, p1_bit);
        let opp_flips2 = flips_for_move_unchecked(opp, me, p2_bit);
        let opp_flips3 = flips_for_move_unchecked(opp, me, p3_bit);

        if opp_flips0 == 0 && opp_flips1 == 0 && opp_flips2 == 0 && opp_flips3 == 0 {
            return (game_over_scaled_from_bits(me, opp), PASS);
        }

        let (opp_score, _mv) = exact_last4(opp, me, empties);
        return (-opp_score, PASS);
    }

    let mut best: Score = SCORE_MIN;
    let mut best_move: Move = PASS;

    if flips0 != 0 {
        let new_me = me | p0_bit | flips0;
        let new_opp = opp & !flips0;
        let (child, _mv) = exact_last3(new_opp, new_me, empties ^ p0_bit);
        let score = -child;

        best = score;
        best_move = p0_bit.trailing_zeros() as Move;
    }
    if flips1 != 0 {
        let new_me = me | p1_bit | flips1;
        let new_opp = opp & !flips1;
        let (child, _mv) = exact_last3(new_opp, new_me, empties ^ p1_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p1_bit.trailing_zeros() as Move;
        }
    }
    if flips2 != 0 {
        let new_me = me | p2_bit | flips2;
        let new_opp = opp & !flips2;
        let (child, _mv) = exact_last3(new_opp, new_me, empties ^ p2_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p2_bit.trailing_zeros() as Move;
        }
    }
    if flips3 != 0 {
        let new_me = me | p3_bit | flips3;
        let new_opp = opp & !flips3;
        let (child, _mv) = exact_last3(new_opp, new_me, empties ^ p3_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p3_bit.trailing_zeros() as Move;
        }
    }

    (best, best_move)
}

/// Exact solve with last 5 empties.
#[inline(always)]
fn exact_last5(me: u64, opp: u64, empties: u64) -> (Score, Move) {
    debug_assert!(empties.count_ones() == 5);

    let mut e = empties;

    let p0_bit = e & e.wrapping_neg();
    e ^= p0_bit;
    let p1_bit = e & e.wrapping_neg();
    e ^= p1_bit;
    let p2_bit = e & e.wrapping_neg();
    e ^= p2_bit;
    let p3_bit = e & e.wrapping_neg();
    e ^= p3_bit;
    let p4_bit = e;

    let flips0 = flips_for_move_unchecked(me, opp, p0_bit);
    let flips1 = flips_for_move_unchecked(me, opp, p1_bit);
    let flips2 = flips_for_move_unchecked(me, opp, p2_bit);
    let flips3 = flips_for_move_unchecked(me, opp, p3_bit);
    let flips4 = flips_for_move_unchecked(me, opp, p4_bit);

    if flips0 == 0 && flips1 == 0 && flips2 == 0 && flips3 == 0 && flips4 == 0 {
        let opp_flips0 = flips_for_move_unchecked(opp, me, p0_bit);
        let opp_flips1 = flips_for_move_unchecked(opp, me, p1_bit);
        let opp_flips2 = flips_for_move_unchecked(opp, me, p2_bit);
        let opp_flips3 = flips_for_move_unchecked(opp, me, p3_bit);
        let opp_flips4 = flips_for_move_unchecked(opp, me, p4_bit);

        if opp_flips0 == 0
            && opp_flips1 == 0
            && opp_flips2 == 0
            && opp_flips3 == 0
            && opp_flips4 == 0
        {
            return (game_over_scaled_from_bits(me, opp), PASS);
        }

        let (opp_score, _mv) = exact_last5(opp, me, empties);
        return (-opp_score, PASS);
    }

    let mut best: Score = SCORE_MIN;
    let mut best_move: Move = PASS;

    if flips0 != 0 {
        let new_me = me | p0_bit | flips0;
        let new_opp = opp & !flips0;
        let (child, _mv) = exact_last4(new_opp, new_me, empties ^ p0_bit);
        let score = -child;

        best = score;
        best_move = p0_bit.trailing_zeros() as Move;
    }
    if flips1 != 0 {
        let new_me = me | p1_bit | flips1;
        let new_opp = opp & !flips1;
        let (child, _mv) = exact_last4(new_opp, new_me, empties ^ p1_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p1_bit.trailing_zeros() as Move;
        }
    }
    if flips2 != 0 {
        let new_me = me | p2_bit | flips2;
        let new_opp = opp & !flips2;
        let (child, _mv) = exact_last4(new_opp, new_me, empties ^ p2_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p2_bit.trailing_zeros() as Move;
        }
    }
    if flips3 != 0 {
        let new_me = me | p3_bit | flips3;
        let new_opp = opp & !flips3;
        let (child, _mv) = exact_last4(new_opp, new_me, empties ^ p3_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p3_bit.trailing_zeros() as Move;
        }
    }
    if flips4 != 0 {
        let new_me = me | p4_bit | flips4;
        let new_opp = opp & !flips4;
        let (child, _mv) = exact_last4(new_opp, new_me, empties ^ p4_bit);
        let score = -child;

        if best_move == PASS || score > best {
            best = score;
            best_move = p4_bit.trailing_zeros() as Move;
        }
    }

    (best, best_move)
}


/// Optional search limits used by Stage4 budgeted search / analysis.
///
/// When `max_nodes` is `None`, search is unbounded.
/// When the budget is exhausted, the search sets an internal `abort` flag and
/// unwinds without storing incomplete TT entries.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchLimits {
    pub max_nodes: Option<u64>,
}

/// Result of a budgeted search call.
///
/// This is used by Stage 5's derivative-style tree scheduler to account for
/// work precisely (node visits) and to distinguish a genuine score from a
/// budget abort.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchOutcome {
    pub score: Score,
    pub best_move: Move,
    pub nodes: u64,
    pub aborted: bool,
}

/// Stop policy for iterative Top-N analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopPolicy {
    /// Converge to the same result as fixed-depth analysis (no heuristic early stop),
    /// unless the node budget is exhausted.
    Complete,
    /// Allow early stop based on a progress metric (gap shrink rate) and the node budget.
    GoodStop,
}

/// Why the iterative analysis stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Complete,
    GoodStop,
    Budget,
    NoMoves,
    MaxRounds,
}

#[derive(Clone, Copy, Debug)]
pub struct AnalyzeIterStats {
    pub nodes_used: u64,
    pub rounds: u32,
    /// Current "top-N boundary gap" in Score units:
    /// `gap = max_outside_upper - nth_inside_score`.
    pub gap: Score,
    pub complete: bool,
    pub stop_reason: StopReason,
}

#[derive(Clone, Debug)]
pub struct AnalyzeIterResult {
    /// Sorted by score descending (from current side-to-move perspective).
    pub pairs: Vec<(Move, Score)>,
    pub stats: AnalyzeIterStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalyzeMode {
    Midgame { depth: u8 },
    Exact,
}


// -----------------------------------------------------------------------------
// Stage 6: Unified analyze_top_n facade (compat layer)
// -----------------------------------------------------------------------------

/// Default `seed_depth` for the Stage6 analysis facade.
///
/// `0` means "auto" (use DerivativeConfig's adaptive 2..5 selection).
pub const DEFAULT_SEED_DEPTH: u8 = 0;

/// Default initial aspiration half-window for alpha-beta root searches.
///
/// This is expressed in Sonetto `Score` units (`SCALE == 32` per disc).
///
/// Sensei mapping: aspiration window in `EvaluatorAlphaBeta`.
pub const DEFAULT_ASPIRATION_WIDTH: Score = 4 * SCALE; // ~= 4 discs

/// Default node budget used by budgeted modes (iterative / derivative).
///
/// Sensei mapping: `max_proof`-style node budget / abort condition.
pub const DEFAULT_NODE_BUDGET: u64 = 1_000_000;

/// Default maximum number of tree nodes for the derivative scheduler arena.
///
/// Sensei mapping: `TreeNodeSupplier` arena capacity.
pub const DEFAULT_TREE_NODE_CAP: usize = 120_000;

/// High-level strategy used by [`Searcher::analyze_top_n`].
///
/// This is the **Stage 6 compatibility layer** requested in `plan.txt`: it allows
/// multiple *analysis* schedulers to coexist behind a single entry point.
///
/// Sensei mapping (names preserved for easier cross-reading):
/// - [`AnalyzeTopNStrategy::Fixed`]      → Sensei `EvaluatorAlphaBeta` (normal αβ)
/// - [`AnalyzeTopNStrategy::Iterative`]  → Sensei Derivative-style *Progress/Advancement* loop
/// - [`AnalyzeTopNStrategy::Derivative`] → Sensei `EvaluatorDerivative` (best-first tree scheduler)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalyzeTopNStrategy {
    /// Fixed-depth alpha-beta analysis.
    ///
    /// Equivalent to calling:
    /// - [`Searcher::analyze_top_n_mid`]  when `mode = Midgame{..}`
    /// - [`Searcher::analyze_top_n_exact`] when `mode = Exact`
    Fixed,

    /// Stage 4 iterative Top-N analysis.
    ///
    /// This is Sonetto's port of Sensei's Derivative *Progress/Advancement* idea:
    /// we maintain per-move bounds and advance one critical move at a time until
    /// either:
    /// - we converge (`StopPolicy::Complete`), or
    /// - we hit the node budget / good-stop heuristic (`StopPolicy::GoodStop`).
    Iterative,

    /// Stage 5 derivative best-first tree scheduler (single-threaded).
    ///
    /// This is the port of Sensei `evaluatederivative` (best-first leaf selection
    /// + weak bounds + budgeted exact proof fallback).
    Derivative,
}

/// Default/tunable parameters shared by multiple analysis modes.
///
/// Required by Stage 6:
/// - `seed_depth`
/// - `aspiration_width`
/// - `node_budget`
/// - `tree_node_cap`
///
/// Sensei mapping:
/// - `seed_depth`        → `EvaluatorThread::AddChildren` seeding depth (derivative)
/// - `aspiration_width`  → root aspiration half-window used by `EvaluatorAlphaBeta`
/// - `node_budget`       → Sensei's *max_proof* / early-abort budget (Stage 4/5)
/// - `tree_node_cap`     → `TreeNodeSupplier` fixed arena capacity (derivative)
///
/// Notes:
/// - `aspiration_width` is expressed in Sonetto [`Score`] units (`SCALE=32` per disc).
/// - `seed_depth = 0` means **auto** (keep Stage 5 adaptive seeding, typically 2..5 ply).
/// - `node_budget = None` means **unbounded** (not recommended for Wasm/UI).
#[derive(Clone, Copy, Debug)]
pub struct AnalyzeTopNParams {
    /// Derivative seeding depth (0 = auto/adaptive).
    pub seed_depth: u8,
    /// Initial aspiration half-window in `Score` units (e.g. `4*SCALE`).
    pub aspiration_width: Score,
    /// Node budget used by iterative/derivative schedulers.
    pub node_budget: Option<u64>,
    /// Derivative arena cap (max tree nodes).
    pub tree_node_cap: usize,
}

/// Default parameters tuned for **interactive analysis** (Wasm-friendly).
///
/// These defaults are intentionally conservative; native callers can raise them.
impl Default for AnalyzeTopNParams {
    fn default() -> Self {
        Self {
            // 0 = keep Stage5 adaptive logic (Sensei-style).
            seed_depth: DEFAULT_SEED_DEPTH,
            // ~= 4 discs when SCALE=32. Matches Stage4/5 root aspiration defaults.
            aspiration_width: DEFAULT_ASPIRATION_WIDTH,
            // Budget chosen to keep UI responsive but still allow meaningful work.
            node_budget: Some(DEFAULT_NODE_BUDGET),
            // Matches Stage5 DerivativeConfig default.
            tree_node_cap: DEFAULT_TREE_NODE_CAP,
        }
    }
}

/// Unified Top-N analysis request.
///
/// This is the single entry point requested in Stage 6:
/// call [`Searcher::analyze_top_n`] with this struct to select *both* the base
/// evaluation mode (midgame vs exact) and the scheduler strategy (fixed vs
/// iterative vs derivative).
#[derive(Clone, Copy, Debug)]
pub struct AnalyzeTopNRequest {
    /// Base evaluation semantics:
    /// - midgame fixed-depth αβ, or
    /// - exact solve-to-end.
    pub mode: AnalyzeMode,

    /// Number of moves to report (clamped to legal move count).
    pub top_n: usize,

    /// Which scheduler/analysis engine to use.
    pub strategy: AnalyzeTopNStrategy,

    /// Shared tunables (defaults provided by [`AnalyzeTopNParams::default`]).
    pub params: AnalyzeTopNParams,

    /// Only used when `strategy == Iterative`.
    pub stop_policy: StopPolicy,
}

impl AnalyzeTopNRequest {
    /// Create a request with sensible defaults for interactive use.
    #[inline]
    pub fn new(mode: AnalyzeMode, top_n: usize) -> Self {
        Self {
            mode,
            top_n,
            strategy: AnalyzeTopNStrategy::Fixed,
            params: AnalyzeTopNParams::default(),
            stop_policy: StopPolicy::GoodStop,
        }
    }

    /// Convenience: set strategy.
    #[inline]
    pub fn with_strategy(mut self, strategy: AnalyzeTopNStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Convenience: override the shared tunables.
    #[inline]
    pub fn with_params(mut self, params: AnalyzeTopNParams) -> Self {
        self.params = params;
        self
    }

    /// Convenience: override the iterative stop policy.
    #[inline]
    pub fn with_stop_policy(mut self, stop_policy: StopPolicy) -> Self {
        self.stop_policy = stop_policy;
        self
    }
}

/// Mode-dependent statistics produced by [`Searcher::analyze_top_n`].
#[derive(Clone, Debug)]
pub enum AnalyzeTopNStats {
    /// Fixed-depth analysis has only a node count and an abort flag.
    Fixed { nodes_used: u64, aborted: bool },
    /// Stage 4 iterative analysis reports progress details.
    Iterative(AnalyzeIterStats),
    /// Stage 5 derivative scheduler returns its full root summary.
    Derivative(crate::derivative::DerivativeResult),
}

/// Unified Top-N analysis output.
#[derive(Clone, Debug)]
pub struct AnalyzeTopNResult {
    /// Top-N `(move, score)` pairs, sorted by score descending from the current
    /// side-to-move perspective.
    pub pairs: Vec<(Move, Score)>,

    /// Mode-specific statistics / diagnostics.
    pub stats: AnalyzeTopNStats,
}


// -----------------------------------------------------------------------------
// Stage 3: stability-based alpha-beta pruning (Sensei UseStabilityCutoff)
// -----------------------------------------------------------------------------

/// Sensei `UseStabilityCutoff(depth)` equivalent.
///
/// In Sensei this is `return depth > 3;`.
#[inline(always)]
const fn use_stability_cutoff(depth: u8) -> bool {
    depth > 3
}

/// When a stability upper bound is very close to `alpha`, the node is unlikely
/// to improve the window; we can switch to cheaper move ordering.
///
/// Sensei uses a constant ~120 in its internal scale. In Sonetto's `Score` scale
/// (`SCALE=32`), ~4 discs corresponds to 128.
const UNLIKELY_STABILITY_MARGIN: Score = 128; // ~= 4 discs

// -----------------------------------------------------------------------------
// Root ordering / aspiration tuning (Stage 1)
// -----------------------------------------------------------------------------

/// Root seed: mobility term weight in Score units.
///
/// Intentionally small (only affects root move ordering, never the final
/// fixed-depth score).
const ROOT_SEED_MOB_WEIGHT: Score = 8; // ~= 1/4 disc (SCALE=32)

/// Small root ordering biases (tie-breakers).
///
/// These are deliberately tiny compared to typical eval/search score ranges.
const ROOT_TT_BONUS: Score = 64; // ~= 2 discs
const ROOT_TT2_BONUS: Score = 32; // ~= 1 disc
const ROOT_KILLER1_BONUS: Score = 16;
const ROOT_KILLER2_BONUS: Score = 8;

/// Aspiration window initial half-width in Score units.
///
/// Keep this moderately wide to avoid frequent fail-high/low re-searches.

/// Depth tag bias for endgame exact solver TT entries.
///
/// We reuse a single TT for midgame + exact endgame. To avoid cross-contamination,
/// exact-solver entries store depth as `TT_DEPTH_EXACT_BIAS + empty_count` (>=128),
/// while midgame search depths stay <128.
const TT_DEPTH_EXACT_BIAS: u8 = 128;

// -----------------------------------------------------------------------------
// Move ordering iterators (Stage 2, Sensei-style)
// -----------------------------------------------------------------------------

/// Move ordering strategy.
///
/// These correspond (semantically) to Sensei's `MoveIterator*` family:
/// - `VeryQuick`  -> `MoveIteratorVeryQuick`
/// - `Quick`      -> `MoveIteratorQuick<true/false>` (simplified buckets)
/// - `MinOppMoves`-> `MoveIteratorMinimizeOpponentMoves`
/// - `Disproof`   -> `MoveIteratorDisproofNumber` (simplified heuristic)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderMode {
    VeryQuick,
    Quick,
    MinOppMoves,
    DisproofNumber,
}

/// If a position looks *unlikely* to raise `alpha`, we switch to cheaper ordering.
///
/// Sensei gates this behaviour to moderately shallow depths; we do the same to
/// avoid harming deep PV stability.
const UNLIKELY_MAX_DEPTH: u8 = 13;

/// If our fast depth-0 estimate is below `alpha` by at least this margin, we
/// consider the node "unlikely".
const UNLIKELY_EVAL_MARGIN: Score = 64; // ~= 2 discs

/// In very tight windows, we tolerate a smaller margin.
const UNLIKELY_EVAL_MARGIN_TIGHT: Score = 32; // ~= 1 disc

/// Window half-width threshold considered "tight".
const TIGHT_WINDOW: Score = 32; // ~= 1 disc

/// Disproof-number mode is only worthwhile once the remaining empties are large
/// enough (mirrors Sensei's kMinEmptiesForDisproofNumber = 12).
const MIN_EMPTIES_FOR_DISPROOF_NUMBER: u8 = 12;

/// In scored iterators, history is only a tie-breaker. We therefore downscale
/// it to avoid overwhelming the primary heuristic.
const HISTORY_SHIFT: u32 = 4; // /16

/// Sensei's `kSquareValue` table (small tie-breaker / locality bias).
///
/// The values are symmetric, so they are robust to coordinate conventions as
/// long as we stay in row-major `bitpos` indexing.
const SQUARE_VALUE: [i32; 64] = [
    18, 4, 16, 12, 12, 16, 4, 18,
    4, 2, 6, 8, 8, 6, 2, 4,
    16, 6, 14, 10, 10, 14, 6, 16,
    12, 8, 10, 0, 0, 10, 8, 12,
    12, 8, 10, 0, 0, 10, 8, 12,
    16, 6, 14, 10, 10, 14, 6, 16,
    4, 2, 6, 8, 8, 6, 2, 4,
    18, 4, 16, 12, 12, 16, 4, 18,
];

// --- Static square buckets (Quick mode) ---

const CORNER_MASK: u64 = (1u64 << 0) | (1u64 << 7) | (1u64 << 56) | (1u64 << 63);

/// X-squares (diagonal-adjacent to corners).
const X_MASK: u64 = (1u64 << 9) | (1u64 << 14) | (1u64 << 49) | (1u64 << 54);

/// C-squares (orthogonally-adjacent to corners).
const C_MASK: u64 = (1u64 << 1)
    | (1u64 << 6)
    | (1u64 << 8)
    | (1u64 << 15)
    | (1u64 << 48)
    | (1u64 << 55)
    | (1u64 << 57)
    | (1u64 << 62);

// Per-corner near squares (C and X) used for dynamic danger classification.
//
// These squares are only "danger" when the corresponding corner is empty.
// When a corner is already occupied, X/C are much less pathological and
// should not be artificially delayed (important when using LMR).
const A1_NEAR: u64 = (1u64 << 1) | (1u64 << 8) | (1u64 << 9);
const H1_NEAR: u64 = (1u64 << 6) | (1u64 << 15) | (1u64 << 14);
const A8_NEAR: u64 = (1u64 << 57) | (1u64 << 48) | (1u64 << 49);
const H8_NEAR: u64 = (1u64 << 62) | (1u64 << 55) | (1u64 << 54);

#[inline(always)]
fn danger_mask_from_occupied(occupied: u64) -> u64 {
    let mut m: u64 = 0;
    if (occupied & (1u64 << 0)) == 0 {
        m |= A1_NEAR;
    }
    if (occupied & (1u64 << 7)) == 0 {
        m |= H1_NEAR;
    }
    if (occupied & (1u64 << 56)) == 0 {
        m |= A8_NEAR;
    }
    if (occupied & (1u64 << 63)) == 0 {
        m |= H8_NEAR;
    }
    m
}

/// Board edges (including corners).
const EDGE_MASK: u64 = 0xFF818181818181FF;

/// "Safe" edges: edges excluding corners and C-squares.
const EDGE_SAFE_MASK: u64 = EDGE_MASK & !CORNER_MASK & !C_MASK;

/// Center: all non-edge squares excluding X-squares.
const CENTER_MASK: u64 = (!EDGE_MASK) & !X_MASK;

#[inline(always)]
fn neighbors(bb: u64) -> u64 {
    // 8-neighborhood, with file masks to avoid wrap-around.
    let a = bb & NOT_A;
    let h = bb & NOT_H;

    (bb << 8)
        | (bb >> 8)
        | (h << 1)
        | (a >> 1)
        | (h << 9)
        | (h >> 7)
        | (a << 7)
        | (a >> 9)
}

#[inline(always)]
fn unique_set(b: u64) -> u64 {
    if b.count_ones() == 1 { b } else { 0 }
}

#[inline(always)]
fn first_last_set(b: u64) -> u64 {
    if b == 0 {
        0
    } else {
        let lo = 1u64 << b.trailing_zeros();
        let hi = 1u64 << (63 - b.leading_zeros());
        lo | hi
    }
}

// Sensei helpers (ported): dynamic edge masks based on current empties.
#[inline(always)]
fn unique_in_edges(empties: u64) -> u64 {
    const FIRST_ROW: u64 = 0x0000_0000_0000_00FF;
    const LAST_ROW: u64 = 0xFF00_0000_0000_0000;
    const FIRST_COL: u64 = 0x0101_0101_0101_0101;
    const LAST_COL: u64 = 0x8080_8080_8080_8080;

    unique_set(empties & FIRST_ROW)
        | unique_set(empties & LAST_ROW)
        | unique_set(empties & FIRST_COL)
        | unique_set(empties & LAST_COL)
}

#[inline(always)]
fn first_last_in_edges(empties: u64) -> u64 {
    const FIRST_ROW: u64 = 0x0000_0000_0000_00FF;
    const LAST_ROW: u64 = 0xFF00_0000_0000_0000;
    const FIRST_COL: u64 = 0x0101_0101_0101_0101;
    const LAST_COL: u64 = 0x8080_8080_8080_8080;

    (first_last_set(empties & FIRST_ROW)
        | first_last_set(empties & LAST_ROW)
        | first_last_set(empties & FIRST_COL)
        | first_last_set(empties & LAST_COL))
        & !CORNER_MASK
}

#[inline(always)]
fn is_unlikely(eval0: Score, alpha: Score, beta: Score) -> bool {
    if eval0 <= alpha.saturating_sub(UNLIKELY_EVAL_MARGIN) {
        return true;
    }
    let window = beta.saturating_sub(alpha);
    if window <= TIGHT_WINDOW && eval0 <= alpha.saturating_sub(UNLIKELY_EVAL_MARGIN_TIGHT) {
        return true;
    }
    false
}

#[inline(always)]
fn select_order_mode(depth_for_mode: u8, solve: bool, unlikely: bool) -> OrderMode {
    // Sensei only uses the "unlikely" shortcut up to a certain depth.
    let unlikely = unlikely && depth_for_mode <= UNLIKELY_MAX_DEPTH;

    if solve {
        if unlikely {
            // Mirroring Sensei: extremely cheap ordering when we likely fail-low.
            if depth_for_mode <= 9 {
                return OrderMode::VeryQuick;
            }
            return OrderMode::Quick;
        }

        // Endgame solve ordering ladder.
        if depth_for_mode <= 9 {
            return OrderMode::Quick;
        }
        if depth_for_mode < MIN_EMPTIES_FOR_DISPROOF_NUMBER {
            return OrderMode::MinOppMoves;
        }
        return OrderMode::DisproofNumber;
    }

    // Midgame ordering ladder.
    if unlikely {
        if depth_for_mode <= 4 {
            return OrderMode::VeryQuick;
        }
        return OrderMode::Quick;
    }

    if depth_for_mode <= 4 {
        OrderMode::Quick
    } else {
        OrderMode::MinOppMoves
    }
}

/// Searcher holds TT + heuristics + per-ply scratch buffers (no recursion allocation).
pub struct Searcher {
    /// Main transposition table for this searcher.
    pub tt: TranspositionTable,

    /// Configured TT size (in megabytes). This is tracked so that parallel
    /// root-split search can spawn worker searchers with a sensible per-thread
    /// TT budget.
    tt_mb: usize,

    // Eval context
    pub weights: Weights,
    pub feats: FeatureDefs,
    pub swap: SwapTables,
    pub occ: OccMap,

    // P1-1: tiny local TT for endgame solves (cache-hot, no atomics)
    end_ltt: EndLocalTT,

    killers: [[Move; 2]; MAX_PLY],
    history: [[u32; 64]; 2],

    // per-ply move buffers
    move_buf: [[Move; MAX_MOVES]; MAX_PLY],
    score_buf: [[i32; MAX_MOVES]; MAX_PLY],

    // P1-1: per-ply cached flips for move-ordering modes that already compute flips.
    //
    // IMPORTANT: Only some ordering modes fill this buffer. See `flips_valid`.
    flips_buf: [[u64; MAX_MOVES]; MAX_PLY],

    /// Whether `flips_buf[ply][..mc]` is valid for the current node at `ply`.
    ///
    /// Many ordering modes (VeryQuick/Quick) do **not** compute flips; in that
    /// case we must treat the cached flips as 0 without paying an O(mc) clear.
    flips_valid: [bool; MAX_PLY],

    // P0-7: per-ply Undo slots reused across moves (avoid Undo::default in hot loops)
    undo_buf: [Undo; MAX_PLY],

    // P0-9: scratch buffers for stable parity partitioning in exact endgame
    parity_tmp_moves: [Move; MAX_MOVES],
    parity_tmp_flips: [u64; MAX_MOVES],

    pub nodes: u64,

    // --- Stage 4: budgeted abort support (node budget) ---
    limits: SearchLimits,
    abort: bool,

    // P1-2: ProbCut probe guard (disable nested ProbCut probes).
    probcut_nesting: u8,

    // P5-1: cached derivative evaluator (arena reuse, avoids large per-call allocations).
    derivative_cache: Option<crate::derivative::DerivativeEvaluator>,
}

impl Searcher {
    pub fn new(tt_mb: usize, weights: Weights, feats: FeatureDefs, swap: SwapTables, occ: OccMap) -> Self {
        // Keep the stored size consistent with `TranspositionTable::new` (which
        // already guards against 0-sized tables).
        let tt_mb = tt_mb.max(1);
        Self {
            tt: TranspositionTable::new(tt_mb),
            tt_mb,
            weights,
            feats,
            swap,
            occ,
            end_ltt: EndLocalTT::new(),
            killers: [[PASS; 2]; MAX_PLY],
            history: [[0u32; 64]; 2],
            move_buf: [[PASS; MAX_MOVES]; MAX_PLY],
            score_buf: [[0i32; MAX_MOVES]; MAX_PLY],
            flips_buf: [[0u64; MAX_MOVES]; MAX_PLY],
            flips_valid: [false; MAX_PLY],
            undo_buf: [Undo::default(); MAX_PLY],
            parity_tmp_moves: [PASS; MAX_MOVES],
            parity_tmp_flips: [0u64; MAX_MOVES],
            nodes: 0,
            limits: SearchLimits::default(),
            abort: false,
            probcut_nesting: 0,
            derivative_cache: None,
        }
    }

    /// Return the configured transposition table size (in MB).
    ///
    /// This is primarily used by parallel helpers to split TT memory across
    /// worker searchers.
    #[inline]
    pub fn tt_mb(&self) -> usize {
        self.tt_mb
    }

    /// Entry point: returns (score, best_move).
    #[inline(always)]
    pub fn search(&mut self, board: &mut Board, alpha: Score, beta: Score, depth: u8) -> (Score, Move) {
        self.nodes = 0;
        self.clear_abort();

        // Clamp to buffer depth to avoid any OOB on adversarial inputs.
        let depth = depth.min(MAX_PLY as u8);

        // Root init: ensure feature ids match the current bitboards.
        // This is *not* done in recursive nodes.
        recompute_features_in_place(board, &self.occ);

        self.negamax(board, depth, alpha, beta, 0, true)
    }

    /// Nodes visited in the most recent search/analyze call.
    #[inline(always)]
    pub fn last_nodes(&self) -> u64 {
        self.nodes
    }


    /// Parallel **root-split** search (Stage P2-2).
    ///
    /// This is the safest/lowest-risk form of parallelism: only the root move
    /// list is split across threads. Each worker searches a subset of root moves
    /// with its own (thread-local) transposition table and heuristics.
    ///
    /// Why root-split first?
    /// - It matches the checklist guidance ("先做 root split，再考虑 YBWC/PV-split").
    /// - It avoids complex shared-state races inside the recursive search.
    ///
    /// Notes:
    /// - If `threads <= 1`, or the position has <= 1 legal move, this falls back
    ///   to the normal single-thread [`Searcher::search`].
    /// - If a node budget is installed (`limits.max_nodes.is_some()`), we also
    ///   fall back to single-threaded search to preserve budget semantics.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn search_root_split(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        depth: u8,
        threads: usize,
    ) -> (Score, Move) {
        // Root bookkeeping matches `search`.
        self.nodes = 0;
        self.clear_abort();

        // Clamp to buffer depth to avoid any OOB on adversarial inputs.
        let depth = depth.min(MAX_PLY as u8);

        // If the caller requested a strict node budget, keep semantics identical
        // to the single-thread search (root-split would otherwise overrun).
        if self.limits.max_nodes.is_some() {
            return self.search(board, alpha, beta, depth);
        }

        // Shallow depths are not worth threading (and `depth-1` would underflow at 0).
        if threads <= 1 || depth <= 1 {
            return self.search(board, alpha, beta, depth);
        }

        // Root init: ensure feature ids match the current bitboards.
        recompute_features_in_place(board, &self.occ);

        // Generate root moves.
        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);

        // PASS / terminal handling stays in the normal negamax.
        if mask == 0 {
            return self.negamax(board, depth, alpha, beta, 0, true);
        }

        let mut root_moves: [Move; MAX_MOVES] = [PASS; MAX_MOVES];
        let mc = push_moves_from_mask(mask, &mut root_moves);

        // No benefit if there is only one move.
        if mc <= 1 {
            return self.negamax(board, depth, alpha, beta, 0, true);
        }

        // Cap threads by the number of moves (and avoid 0).
        let t = threads.max(1).min(mc);
        if t <= 1 {
            return self.negamax(board, depth, alpha, beta, 0, true);
        }

        // Keep total TT memory roughly constant across workers.
        let worker_tt_mb: usize = (self.tt_mb / t).max(1);

        // Shared eval context clones (cheap) for worker construction.
        let weights = self.weights.clone();
        let feats = self.feats.clone();
        let swap = self.swap.clone();
        let occ = self.occ.clone();

        // Clone the board once for distribution. Each thread keeps a local copy
        // and applies/undos root moves sequentially (no per-move cloning).
        let base_board = board.clone();

        let child_depth = depth - 1;
        let alpha_child = -beta;
        let beta_child = -alpha;

        // Spawn workers for thread ids 1..t-1.
        let mut handles = Vec::with_capacity(t.saturating_sub(1));
        for tid in 1..t {
            let mut worker = Searcher::new(
                worker_tt_mb,
                weights.clone(),
                feats.clone(),
                swap.clone(),
                occ.clone(),
            );
            // Keep worker abort semantics clean.
            worker.clear_abort();
            worker.nodes = 0;

            let mut local_board = base_board.clone();
            let local_moves = root_moves; // Copy (Move=u8)
            let handle = std::thread::spawn(move || -> (Score, usize, Move, u64) {
                let mut best_score: Score = -INF;
                let mut best_idx: usize = usize::MAX;
                let mut best_move: Move = PASS;

                for idx in (tid..mc).step_by(t) {
                    let mv = local_moves[idx];

                    let mut u = Undo::default();
                    let ok = local_board.apply_move_with_occ(mv, &mut u, Some(&worker.occ));
                    debug_assert!(ok);
                    if !ok {
                        continue;
                    }

                    let (child, _pv_mv) = worker.negamax(
                        &mut local_board,
                        child_depth,
                        alpha_child,
                        beta_child,
                        1,
                        true,
                    );

                    local_board.undo_move_with_occ(&u, Some(&worker.occ));

                    // On any unexpected abort (should not happen here), stop and
                    // report the best move found so far.
                    if worker.abort {
                        break;
                    }

                    let score = -child;
                    if score > best_score || (score == best_score && idx < best_idx) {
                        best_score = score;
                        best_idx = idx;
                        best_move = mv;
                    }
                }

                (best_score, best_idx, best_move, worker.nodes)
            });

            handles.push(handle);
        }

        // Main thread acts as worker tid=0 using the existing searcher (reuses its TT).
        let mut main_board = base_board.clone();
        let mut best_score: Score = -INF;
        let mut best_idx: usize = usize::MAX;
        let mut best_move: Move = PASS;

        for idx in (0..mc).step_by(t) {
            let mv = root_moves[idx];

            let mut u = Undo::default();
            let ok = main_board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
            debug_assert!(ok);
            if !ok {
                continue;
            }

            let (child, _pv_mv) = self.negamax(
                &mut main_board,
                child_depth,
                alpha_child,
                beta_child,
                1,
                true,
            );

            main_board.undo_move_with_occ(&u, Some(&self.occ));

            if self.abort {
                break;
            }

            let score = -child;
            if score > best_score || (score == best_score && idx < best_idx) {
                best_score = score;
                best_idx = idx;
                best_move = mv;
            }
        }

        // Collect worker results + sum node counts.
        let mut total_nodes = self.nodes;
        for h in handles {
            if let Ok((s, idx, mv, nodes)) = h.join() {
                total_nodes = total_nodes.wrapping_add(nodes);
                if s > best_score || (s == best_score && idx < best_idx) {
                    best_score = s;
                    best_idx = idx;
                    best_move = mv;
                }
            }
        }
        self.nodes = total_nodes;

        (best_score, best_move)
    }

    /// Wasm fallback for the root-split API: no threads.
    #[cfg(target_arch = "wasm32")]
    pub fn search_root_split(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        depth: u8,
        _threads: usize,
    ) -> (Score, Move) {
        self.search(board, alpha, beta, depth)
    }


    /// Budgeted midgame search wrapper.
    ///
    /// This is a thin adapter around [`Searcher::search`]. It temporarily
    /// installs `limits`, runs the search, and returns a [`SearchOutcome`] that
    /// includes the node count and an explicit `aborted` flag.
    pub fn search_with_limits(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        depth: u8,
        limits: SearchLimits,
    ) -> SearchOutcome {
        // Stage 6 cleanup:
        // Budgeted wrappers must *not* leak the internal `abort` flag into subsequent calls.
        // The only supported way to observe an abort is the returned `SearchOutcome`.
        let old_limits = self.limits;
        let old_abort = self.abort;

        self.limits = limits;
        self.abort = false;

        let (score, best_move) = self.search(board, alpha, beta, depth);
        let out = SearchOutcome {
            score,
            best_move,
            nodes: self.nodes,
            aborted: self.abort,
        };

        // Restore previous state.
        self.limits = old_limits;
        self.abort = old_abort;
        out
    }

    /// Budgeted exact endgame solve wrapper.
    ///
    /// This calls the internal exact solver (empties-to-end), but exposes the
    /// same `limits` mechanism as Stage 4.
    pub fn exact_search_with_limits(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        limits: SearchLimits,
    ) -> SearchOutcome {
        // Save + install limits.
        // Stage 6 cleanup: like `search_with_limits`, this wrapper must not leak `abort`.
        let old_limits = self.limits;
        let old_abort = self.abort;
        self.limits = limits;

        self.nodes = 0;
        self.abort = false;

        let (score, best_move) = self.exact_negamax(board, alpha, beta, 0);
        let out = SearchOutcome {
            score,
            best_move,
            nodes: self.nodes,
            aborted: self.abort,
        };

        // Restore previous state.
        self.limits = old_limits;
        self.abort = old_abort;
        out
    }

    #[inline(always)]
    fn eval(&self, board: &Board) -> Score {
        // `evaluate` returns tanh-mapped [-1000, 1000]. Convert into the engine's Score units.
        eval1000_to_score(evaluate(board, &self.weights, &self.feats, &self.swap))
    }

    // -------------------------------------------------------------------------
    // Stage 4 budget/abort support
    // -------------------------------------------------------------------------

    /// Return true if the current call should abort due to the node budget.
    ///
    /// Important: this function is only consulted at the *entry* of each node,
    /// so a single node may still expand some work before control reaches the
    /// next node check. This keeps the implementation simple and predictable.
    #[inline(always)]
    fn budget_exceeded(&mut self) -> bool {
        if let Some(max_nodes) = self.limits.max_nodes {
            if self.nodes >= max_nodes {
                self.abort = true;
                return true;
            }
        }
        false
    }

    /// Clear any previous abort flag (does not change limits).
    #[inline(always)]
    fn clear_abort(&mut self) {
        self.abort = false;
        self.probcut_nesting = 0;
    }

    // -------------------------------------------------------------------------
    // Stage 2 move ordering (Sensei-style iterators)
    // -------------------------------------------------------------------------

    /// Heuristic bonus used as a *tie-breaker* in scored iterators.
    ///
    /// - TT best move is always prioritized.
    /// - Killer moves are next.
    /// - History is downscaled to avoid overwhelming the mode's primary heuristic.
    #[inline(always)]
    fn heuristic_bonus_scaled(
        &self,
        side_idx: usize,
        mv: Move,
        tt_move: Move,
        tt_move2: Option<Move>,
        ply: usize,
    ) -> i32 {
        if mv == tt_move {
            return 1_000_000;
        }
        if tt_move2.is_some() && Some(mv) == tt_move2 {
            return 900_000;
        }

        let k1 = self.killers[ply][0];
        let k2 = self.killers[ply][1];
        if mv == k1 {
            return 800_000;
        }
        if mv == k2 {
            return 700_000;
        }

        // `mv` is a legal move (0..63).
        (self.history[side_idx][mv as usize] >> HISTORY_SHIFT) as i32
    }

    /// Fill `move_buf[ply]` according to the selected ordering mode.
    ///
    /// This is the Sonetto counterpart of Sensei's `MoveIterator*::Setup(...)`.
    #[inline(always)]
    fn order_moves_for_node(
        &mut self,
        board: &Board,
        me: u64,
        opp: u64,
        mask: u64,
        tt_move: Move,
        tt_move2: Option<Move>,
        ply: usize,
        depth_for_mode: u8,
        alpha: Score,
        beta: Score,
        stability_upper: Option<Score>,
        solve: bool,
    ) -> usize {
        // Default: this node's ordering mode does not provide cached flips.
        // Specific modes that compute flips (MinOppMoves/DisproofNumber) will
        // mark the prefix as valid.
        self.flips_valid[ply] = false;

        // Fast depth-0 estimate (cheap): disc diff + mobility diff.
        let me_mob = mask.count_ones() as Score;
        let opp_mob = legal_moves(opp, me).count_ones() as Score;
        let eval0 = disc_diff_scaled(board, board.side) + (me_mob - opp_mob) * ROOT_SEED_MOB_WEIGHT;

        let mut unlikely = is_unlikely(eval0, alpha, beta);

        // Stage 3: Sensei-style `unlikely` augmentation using the stability upper bound.
        // If even the provable upper bound is only slightly above alpha, the node is
        // very unlikely to tighten the window.
        if let Some(ub) = stability_upper {
            if ub <= alpha.saturating_add(UNLIKELY_STABILITY_MARGIN) {
                unlikely = true;
            }
        }
        let mode = select_order_mode(depth_for_mode, solve, unlikely);

        let side_idx = board.side.idx();

        match mode {
            OrderMode::VeryQuick => self.order_moves_very_quick(me, opp, mask, tt_move, tt_move2, ply),
            OrderMode::Quick => self.order_moves_quick(me, opp, mask, tt_move, tt_move2, ply),
            OrderMode::MinOppMoves => {
                self.order_moves_min_opp_moves(me, opp, mask, tt_move, tt_move2, ply, side_idx)
            }
            OrderMode::DisproofNumber => {
                self.order_moves_disproof_number(me, opp, mask, tt_move, tt_move2, ply, side_idx, beta)
            }
        }
    }

    /// VERY_QUICK: enumerate only cheap candidates (neighbors(opponent) & empty).
    ///
    /// We still intersect with the already-computed legal move mask (`mask`) to
    /// keep correctness and reuse work from move generation.
    #[inline(always)]
    fn order_moves_very_quick(
        &mut self,
        me: u64,
        opp: u64,
        mask: u64,
        tt_move: Move,
        tt_move2: Option<Move>,
        ply: usize,
    ) -> usize {
        let empties = !(me | opp);
        let candidates = neighbors(opp) & empties;

        let mut rem = mask & candidates;
        let out = &mut self.move_buf[ply];
        let mut n: usize = 0;

        #[inline(always)]
        fn push_one(out: &mut [Move; MAX_MOVES], n: &mut usize, rem: &mut u64, mv: Move) {
            if mv == PASS {
                return;
            }
            if mv >= 64 {
                return;
            }
            if *n >= MAX_MOVES {
                return;
            }
            let bit = 1u64 << (mv as u64);
            if (*rem & bit) != 0 {
                out[*n] = mv;
                *n += 1;
                *rem &= !bit;
            }
        }

        // Sensei behaviour: TT best move first.
        push_one(out, &mut n, &mut rem, tt_move);
        if let Some(m2) = tt_move2 {
            push_one(out, &mut n, &mut rem, m2);
        }

        // Then killers (still cheap and often helpful for cutoffs).
        push_one(out, &mut n, &mut rem, self.killers[ply][0]);
        push_one(out, &mut n, &mut rem, self.killers[ply][1]);

        // Remaining moves in stable square order.
        while rem != 0 {
            if n >= MAX_MOVES {
                break;
            }
            let sq = rem.trailing_zeros() as Move;
            out[n] = sq;
            n += 1;
            rem &= rem - 1;
        }

        // This iterator does not compute flips; keep `flips_valid[ply] = false`.

        n
    }

    /// QUICK: static buckets (corner/edge/center/danger), no per-move eval.
    #[inline(always)]
    fn order_moves_quick(
        &mut self,
        me: u64,
        opp: u64,
        mask: u64,
        tt_move: Move,
        tt_move2: Option<Move>,
        ply: usize,
    ) -> usize {
        let occupied = me | opp;
        let empties = !occupied;
        let candidates = neighbors(opp) & empties;

        // Sensei-style dynamic buckets:
        // - X/C squares are only treated as "danger" when the corresponding
        //   corner is empty.
        // - Prioritize "quiet" moves not adjacent to empties.
        let danger = danger_mask_from_occupied(occupied);
        let neigh_me = neighbors(me);
        let safe_interior = (!neighbors(empties)) & neigh_me & !danger;
        let uniq_edge = unique_in_edges(empties) & neigh_me & !danger;
        let first_last_edge = first_last_in_edges(empties) & !danger;
        let edge_safe_dyn = (EDGE_SAFE_MASK | (C_MASK & !danger)) & !CORNER_MASK;
        let center_dyn = CENTER_MASK | (X_MASK & !danger);

        let mut rem = mask & candidates;
        let out = &mut self.move_buf[ply];
        let mut n: usize = 0;

        #[inline(always)]
        fn push_one(out: &mut [Move; MAX_MOVES], n: &mut usize, rem: &mut u64, mv: Move) {
            if mv == PASS {
                return;
            }
            if mv >= 64 {
                return;
            }
            if *n >= MAX_MOVES {
                return;
            }
            let bit = 1u64 << (mv as u64);
            if (*rem & bit) != 0 {
                out[*n] = mv;
                *n += 1;
                *rem &= !bit;
            }
        }

        // TT first.
        push_one(out, &mut n, &mut rem, tt_move);
        if let Some(m2) = tt_move2 {
            push_one(out, &mut n, &mut rem, m2);
        }
        // Killers next.
        push_one(out, &mut n, &mut rem, self.killers[ply][0]);
        push_one(out, &mut n, &mut rem, self.killers[ply][1]);

        // Bucket order (ported/tuned from Sensei):
        //   corners -> quiet interior -> unique edge -> edge endpoints -> safe edges
        //   -> center -> danger -> rest.
        for &bucket in &[
            CORNER_MASK,
            safe_interior,
            uniq_edge,
            first_last_edge,
            edge_safe_dyn,
            center_dyn,
            danger,
        ] {
            let mut bits = rem & bucket;
            while bits != 0 {
                if n >= MAX_MOVES {
                    break;
                }
                let sq = bits.trailing_zeros() as Move;
                out[n] = sq;
                n += 1;
                bits &= bits - 1;
            }
            rem &= !bucket;
        }

        // Should be empty, but keep it correct if our masks ever change.
        while rem != 0 {
            if n >= MAX_MOVES {
                break;
            }
            let sq = rem.trailing_zeros() as Move;
            out[n] = sq;
            n += 1;
            rem &= rem - 1;
        }

        // This iterator does not compute flips; keep `flips_valid[ply] = false`.

        n
    }

    /// MIN_OPP_MOVES: prefer moves that minimize opponent mobility.
    ///
    /// This is analogous to Sensei's `MoveIteratorMinimizeOpponentMoves`.
    fn order_moves_min_opp_moves(
        &mut self,
        me: u64,
        opp: u64,
        mask: u64,
        tt_move: Move,
        tt_move2: Option<Move>,
        ply: usize,
        side_idx: usize,
    ) -> usize {
        let mc = push_moves_from_mask(mask, &mut self.move_buf[ply]);
        for i in 0..mc {
            let mv = self.move_buf[ply][i];
            let mv_bit = 1u64 << (mv as u64);
            let flips = flips_for_move_unchecked(me, opp, mv_bit);

            // P1-1: cache flips so the search loop can reuse it on make.
            self.flips_buf[ply][i] = flips;

            // If this triggers, movegen and flipgen disagree; keep it safe.
            if flips == 0 {
                self.score_buf[ply][i] = i32::MIN / 2;
                continue;
            }

            let new_me = me | mv_bit | flips;
            let new_opp = opp & !flips;

            let opp_moves = legal_moves(new_opp, new_me);
            let opp_cnt = opp_moves.count_ones() as i32;
            let opp_corner_cnt = (opp_moves & CORNER_MASK).count_ones() as i32;

            let base = -((opp_cnt + opp_corner_cnt) * 1000) + SQUARE_VALUE[mv as usize];
            let bonus = self.heuristic_bonus_scaled(side_idx, mv, tt_move, tt_move2, ply);
            self.score_buf[ply][i] = base + bonus;
        }

        sort_moves_desc_with_flips(&mut self.move_buf[ply], &mut self.score_buf[ply], &mut self.flips_buf[ply], mc);
        self.flips_valid[ply] = true;
        mc
    }

    /// DISPROOF_NUMBER (simplified): prefer moves whose cheap estimate is close
    /// to (or above) `beta`, i.e. likely to refute the opponent quickly.
    ///
    /// Stage 4 hook: this keeps the same API surface (`beta`-aware scoring) so
    /// we can later swap in a more faithful proof/disproof-number heuristic.
fn order_moves_disproof_number(
    &mut self,
    me: u64,
    opp: u64,
    mask: u64,
    tt_move: Move,
    tt_move2: Option<Move>,
    ply: usize,
    side_idx: usize,
    beta: Score,
) -> usize {
    // Sensei-style disproof-number move ordering:
    //   value = -DisproofNumberOverProb(child, lower=-beta, approx_eval) + square_value
    // TT move is forced to the front (Sensei uses a large constant).
    //
    // We fill `move_buf[ply]`/`score_buf[ply]` similarly to the other ordering modes.

    let occ = me | opp;
    let empties_before = 64u8 - occ.count_ones() as u8;
    if empties_before == 0 {
        return 0;
    }
    let empties_after = empties_before - 1;

    // Convert Sonetto Score (disc*32) to Sensei EvalLarge (disc*8).
    // Using an arithmetic shift keeps behavior consistent for negative values.
    let upper_eval_large: i32 = (beta >> 2).clamp(-512, 512);
    let lower_eval_large: i32 = -upper_eval_large;

    let mut moves_mask = mask;
    let mut n = 0usize;

    while moves_mask != 0 {
        let idx = moves_mask.trailing_zeros() as u8;
        moves_mask &= moves_mask - 1;
        let mv: Move = idx;
        let mv_bit = 1u64 << idx;

        // `mask` is a legal-move mask; unchecked flip is safe.
        let flips = flips_for_move_unchecked(me, opp, mv_bit);

        // Child position: opponent to move.
        let child_me = opp & !flips;
        let child_opp = me | mv_bit | flips;

        // Depth-1-style approx eval for the child, from the child side-to-move
        // perspective. We call the EGEV2 evaluator's disc-score path.
        let tmp_board = Board {
            player: child_me,
            opponent: child_opp,
            // Slow-path `score_disc` does not depend on `side`.
            side: crate::board::Color::Black,
            empty_count: empties_after,
            hash: 0,
            feat_id_abs: Vec::new(),
            feat_is_pattern_ids: false,
        };
        let approx_eval_large: i32 = crate::eval::score_disc(&tmp_board, &self.weights) * 8;

        let disproof = crate::sensei_extras::endgame_time::disproof_number_over_prob(
            child_me,
            child_opp,
            lower_eval_large,
            approx_eval_large,
        );

        let score: Score = if mv == tt_move {
            // Sensei uses 99,999,999; keep the same order of magnitude.
            100_000_000
        } else {
            let base = -(disproof as Score) + SQUARE_VALUE[mv as usize];
            base + self.heuristic_bonus_scaled(side_idx, mv, tt_move, tt_move2, ply)
        };

        self.move_buf[ply][n] = mv;
        self.score_buf[ply][n] = score;
        self.flips_buf[ply][n] = flips;
        n += 1;
    }

    sort_moves_desc_with_flips(
        &mut self.move_buf[ply],
        &mut self.score_buf[ply],
        &mut self.flips_buf[ply],
        n,
    );
    self.flips_valid[ply] = true;
    n
}



    // ---------------------------------------------------------------------
    // P1-2: Multi-ProbCut / ProbCut
    // ---------------------------------------------------------------------

    #[inline(always)]
    fn mpc_cutoff(
        &mut self,
        board: &mut Board,
        depth: u8,
        alpha: Score,
        beta: Score,
        ply: usize,
        pv: bool,
        me_moves_mask: u64,
    ) -> Option<Score> {
        // Avoid disrupting PV nodes and near-root nodes.
        if pv || ply < PROBCUT_SHALLOW_IGNORE {
            return None;
        }

        // Only makes sense at reasonably large depths.
        if depth < PROBCUT_MIN_DEPTH {
            return None;
        }

        // Don't nest ProbCut probes.
        if self.probcut_nesting != 0 {
            return None;
        }

        // Keep Stage 4 node-budget semantics stable: ProbCut adds extra probes.
        if self.limits.max_nodes.is_some() {
            return None;
        }

        // If there's only one move, just search it normally.
        if me_moves_mask.count_ones() <= 1 {
            return None;
        }

        // Skip if the window is effectively unbounded.
        if beta >= INF / 2 && alpha <= -INF / 2 {
            return None;
        }

        // Cheap disc-diff prefilter to avoid paying for full eval() in almost all nodes.
        let disc = ((board.player.count_ones() as i32) - (board.opponent.count_ones() as i32)) * SCALE;
        let might_high = beta < SCORE_MAX && disc >= beta.saturating_add(PROBCUT_QUICK_TRIGGER);
        let might_low = alpha > SCORE_MIN && disc <= alpha.saturating_sub(PROBCUT_QUICK_TRIGGER);
        if !might_high && !might_low {
            return None;
        }

        let n_discs: i32 = 64 - (board.empty_count as i32);

        let reduction = if depth >= 10 { PROBCUT_REDUCTION_DEEP } else { PROBCUT_REDUCTION_SHALLOW };
        let search_depth = depth.saturating_sub(reduction);
        if search_depth >= depth {
            return None;
        }

        // Compute margins in Score units.
        let err0 = probcut_error_score_depth0(n_discs, depth);
        let err = probcut_error_score_search(n_discs, depth, search_depth);

        // Full static eval (expensive); only reached after quick filter passed.
        let depth0_value = self.eval(board);

        // Fail-high (beta) direction.
        if might_high
            && depth0_value >= beta.saturating_add(err0)
            && beta.saturating_add(err) <= SCORE_MAX
        {
            let probe_beta = beta + err;
            if search_depth == 0 {
                if depth0_value >= probe_beta {
                    return Some(beta);
                }
            } else {
                let probe_alpha = probe_beta - 1;
                self.probcut_nesting += 1;
                let (v, _mv) = self.negamax(board, search_depth, probe_alpha, probe_beta, ply, false);
                self.probcut_nesting -= 1;

                if self.abort {
                    return None;
                }

                if v >= probe_beta {
                    return Some(beta);
                }
            }
        }

        // Fail-low (alpha) direction.
        if might_low
            && depth0_value <= alpha.saturating_sub(err0)
            && alpha.saturating_sub(err) >= SCORE_MIN
        {
            let probe_alpha = alpha - err;
            if search_depth == 0 {
                if depth0_value <= probe_alpha {
                    return Some(alpha);
                }
            } else {
                let probe_beta = probe_alpha + 1;
                self.probcut_nesting += 1;
                let (v, _mv) = self.negamax(board, search_depth, probe_alpha, probe_beta, ply, false);
                self.probcut_nesting -= 1;

                if self.abort {
                    return None;
                }

                if v <= probe_alpha {
                    return Some(alpha);
                }
            }
        }

        None
    }
    fn negamax(
        &mut self,
        board: &mut Board,
        depth: u8,
        mut alpha: Score,
        beta: Score,
        ply: usize,
        pv: bool,
    ) -> (Score, Move) {
        // --- Stage 4: node-budget abort ---
        if self.abort {
            return (0, PASS);
        }
        if self.budget_exceeded() {
            return (0, PASS);
        }
        self.nodes = self.nodes.wrapping_add(1);

        // Safety: fixed per-ply buffers.
        if ply >= MAX_PLY {
            return (self.eval(board), PASS);
        }

        // Leaf
        if depth == 0 {
            return (self.eval(board), PASS);
        }

        // --- Stage 3: stability bound cutoff (Sensei `UseStabilityCutoff`) ---
        // We compute a provable upper bound from (definitely) stable opponent discs.
        // If even that upper bound cannot beat `alpha`, we can fail-low immediately.
        let mut stability_upper: Option<Score> = None;
        #[cfg(feature = "stability_cutoff")]
        {
            if use_stability_cutoff(depth) {
                let (_lo, hi) = stability_bounds_for_side_to_move(board);
                stability_upper = Some(hi);
                if hi <= alpha {
                    return (hi, PASS);
                }
            }
        }

        let alpha_orig = alpha;
        let key = board.hash;

        // --- TT probe ---
        let mut has_tt = false;
        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;

        if let Some(e) = self.tt.probe(key) {
            // Ignore exact-solver TT entries (depth>=128) during midgame search.
            if e.depth < TT_DEPTH_EXACT_BIAS {
                has_tt = true;
                tt_move = e.best_move;
                tt_move2 = e.best_move2;

                if e.depth >= depth {
                    match e.flag {
                        Bound::Exact => return (e.value, e.best_move),
                        Bound::Lower => {
                            if e.value >= beta {
                                return (e.value, e.best_move);
                            }
                        }
                        Bound::Upper => {
                            if e.value <= alpha {
                                return (e.value, e.best_move);
                            }
                        }
                    }
                }
            }
        }

        // --- Move generation ---
        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);

        // No moves: PASS or terminal
        if mask == 0 {
            let opp_mask = legal_moves(opp, me);
            if opp_mask == 0 {
                // Game over: exact disc difference (scaled) from side-to-move.
                return (game_over_scaled(board, board.side), PASS);
            }

            // PASS move
            let undo = {
                let u = &mut self.undo_buf[ply];
                let ok = board.apply_move_with_occ(PASS, u, Some(&self.occ));
                debug_assert!(ok);
                *u
            };

            let (child, _mv) = self.negamax(board, depth - 1, -beta, -alpha, ply + 1, pv);

            board.undo_move_with_occ(&undo, Some(&self.occ));

            // If we hit the node budget deeper in the tree, unwind without using the partial score.
            if self.abort {
                return (0, PASS);
            }

            return (-child, PASS);
        }

        let empty_before = board.empty_count;

        // --- IID (Internal Iterative Deepening) ---
        // Spec condition: d>=4 && !hasTT && mc>1.
        // "Shallow" depth is fixed to 3 plies per spec note.
        let mc_est = mask.count_ones() as usize;

        // --- P1-2: Multi-ProbCut (Egaroucid-inspired) ---
        if let Some(pc) = self.mpc_cutoff(board, depth, alpha, beta, ply, pv, mask) {
            return (pc, PASS);
        }
        if self.abort {
            return (0, PASS);
        }

        if depth >= 4 && !has_tt && mc_est > 1 {
            let (_s, iid_mv) = self.negamax(board, 3, -INF, INF, ply, true);

            // If the budget was exhausted during IID, abort early.
            if self.abort {
                return (0, PASS);
            }

            if iid_mv != PASS {
                tt_move = iid_mv;
            }
        }

        // --- Stage 2 ordering: dynamically pick a Sensei-style iterator ---
        // Midgame depth uses the regular search depth.
        let mc = self.order_moves_for_node(
            board,
            me,
            opp,
            mask,
            tt_move,
            tt_move2,
            ply,
            depth,
            alpha,
            beta,
            stability_upper,
            false, // solve=false (midgame)
        );

        let side_idx = board.side.idx();

        // --- Search loop (PVS + LMR) ---
        let mut best_score: Score = -INF;
        let mut best_move: Move = PASS;
        let mut best2_score: Score = -INF;
        let mut best2_move: Option<Move> = None;

        for i in 0..mc {
            let mv = self.move_buf[ply][i];
            let cached_flips = if self.flips_valid[ply] { self.flips_buf[ply][i] } else { 0 };

            // P0-7: reuse an Undo slot, and avoid checked make-move overhead for legal moves.
            let undo = {
                let u = &mut self.undo_buf[ply];
                if mv != PASS {
                    if cached_flips != 0 {
                        // Safety: `mv` is a legal move and `cached_flips` were computed for this move.
                        unsafe { board.apply_move_with_occ_preflips_unchecked(mv, cached_flips, u, &self.occ) };
                    } else {
                        // Safety: `mv` is a legal move (generated by movegen / ordering).
                        unsafe { board.apply_move_with_occ_unchecked(mv, u, &self.occ) };
                    }
                } else {
                    let ok = board.apply_move_with_occ(PASS, u, Some(&self.occ));
                    debug_assert!(ok);
                }
                *u
            };

            // TT prefetch (Egaroucid/Sensei-style): overlap child-cluster fetch
            // with the remaining per-move bookkeeping before recursion.
            self.tt.prefetch(board.hash);

            let mut child_depth = depth - 1;
            let mut reduced = false;

            // LMR condition align: d>=3 && i>=3 && empty>24
            if depth >= 3 && i >= 3 && empty_before > 24 {
                child_depth = child_depth.saturating_sub(1);
                reduced = true;
            }

            let score: Score;

            if i == 0 {
                // First move: full window
                let (child, _mv2) = self.negamax(board, depth - 1, -beta, -alpha, ply + 1, pv);
                if self.abort {
                    board.undo_move_with_occ(&undo, Some(&self.occ));
                    return (0, PASS);
                }
                score = -child;
            } else {
                // PVS: null-window first
                let a1 = alpha.saturating_add(1);
                let (child, _mv2) = self.negamax(board, child_depth, -a1, -alpha, ply + 1, false);
                if self.abort {
                    board.undo_move_with_occ(&undo, Some(&self.occ));
                    return (0, PASS);
                }
                let mut s = -child;

                // If reduced and improved alpha, re-search at full depth (still null-window).
                if reduced && s > alpha {
                    let (child2, _mv3) = self.negamax(board, depth - 1, -a1, -alpha, ply + 1, false);
                    if self.abort {
                        board.undo_move_with_occ(&undo, Some(&self.occ));
                        return (0, PASS);
                    }
                    s = -child2;
                }

                // If it passes the null window, re-search full window to confirm.
                if s > alpha && s < beta {
                    let (child3, _mv4) = self.negamax(board, depth - 1, -beta, -alpha, ply + 1, true);
                    if self.abort {
                        board.undo_move_with_occ(&undo, Some(&self.occ));
                        return (0, PASS);
                    }
                    s = -child3;
                }

                score = s;
            }

            board.undo_move_with_occ(&undo, Some(&self.occ));

            // Track best / 2nd best (for TT best_move2).
            if score > best_score {
                best2_score = best_score;
                best2_move = if best_move == PASS { None } else { Some(best_move) };

                best_score = score;
                best_move = mv;
            } else if mv != best_move && score > best2_score {
                best2_score = score;
                best2_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
            }

            // Beta cutoff
            if alpha >= beta {
                self.on_cutoff(ply, side_idx, mv, depth);
                break;
            }
        }

        // --- TT store ---
        let flag = if best_score <= alpha_orig {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };

        self.tt.store(key, depth, flag, best_score, best_move, best2_move);

        (best_score, best_move)
    }

    #[inline(always)]
    fn move_score(&self, side_idx: usize, mv: Move, tt_move: Move, tt_move2: Option<Move>, ply: usize) -> i32 {
        if mv == tt_move {
            return 1_000_000;
        }
        if tt_move2.is_some() && Some(mv) == tt_move2 {
            return 900_000;
        }

        let k1 = self.killers[ply][0];
        let k2 = self.killers[ply][1];

        if mv == k1 {
            return 800_000;
        }
        if mv == k2 {
            return 700_000;
        }

        // `mv` is a legal move (0..63).
        self.history[side_idx][mv as usize] as i32
    }

    #[inline(always)]
    fn on_cutoff(&mut self, ply: usize, side_idx: usize, mv: Move, depth: u8) {
        if mv == PASS {
            return;
        }

        // Killer moves (2 slots)
        if self.killers[ply][0] != mv {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = mv;
        }

        // History heuristic
        let d = depth as u32;
        let bonus = d.saturating_mul(d);
        let h = &mut self.history[side_idx][mv as usize];
        *h = h.saturating_add(bonus);
    }
}



impl Searcher {

    // -------------------------------------------------------------------------
    // Root analysis helpers (Top-N) + exact endgame solver
    // -------------------------------------------------------------------------

    /// Extremely cheap root seed score (midgame): static eval + mobility.
    ///
    /// Contract:
    /// - `board` is **after** applying the candidate root move, so `board.side` is the opponent to move.
    /// - Returned score is from the **original root side** perspective.
    #[inline(always)]
    fn root_seed_score_mid_after_move(&self, board: &Board) -> Score {
        // `eval()` is always from side-to-move perspective. After a root move, side-to-move is opponent.
        let base = -self.eval(board);

        // Mobility: (root_side mobility) - (opponent mobility).
        let opp = board.player;
        let me = board.opponent;
        let opp_mob = legal_moves(opp, me).count_ones() as Score;
        let me_mob = legal_moves(me, opp).count_ones() as Score;

        base + (me_mob - opp_mob) * ROOT_SEED_MOB_WEIGHT
    }

    /// Extremely cheap root seed score (exact endgame): disc diff + mobility.
    ///
    /// We intentionally avoid `eval()` here because in the WASM pipeline the board's
    /// pattern feature ids may not be maintained in exact mode.
    ///
    /// Contract:
    /// - `board` is **after** applying the candidate root move, so `board.side` is the opponent to move.
    /// - Returned score is from the **original root side** perspective.
    #[inline(always)]
    fn root_seed_score_exact_after_move(board: &Board) -> Score {
        let root_side = board.side.other();
        let base = disc_diff_scaled(board, root_side);

        let opp = board.player;
        let me = board.opponent;
        let opp_mob = legal_moves(opp, me).count_ones() as Score;
        let me_mob = legal_moves(me, opp).count_ones() as Score;

        base + (me_mob - opp_mob) * ROOT_SEED_MOB_WEIGHT
    }

    /// Root full-depth search for a single move using an aspiration window.
    ///
    /// Contract:
    /// - `board` is **after** applying the candidate root move.
    /// - `depth` is the remaining depth for the child search (i.e. original depth-1).
    /// - Returned score is from the **original root side** perspective.
    fn root_score_mid_with_aspiration(
        &mut self,
        board: &mut Board,
        depth: u8,
        guess_root: Score,
        aspiration_width: Score,
    ) -> Score {
        // Depth-0 is pure eval; aspiration only adds overhead here.
        if depth == 0 {
            return -self.eval(board);
        }

        let mut guess = guess_root;
        let mut window = aspiration_width.max(1);

        // Up to a few iterations; window saturates to full [-INF, INF].
        loop {
            let mut alpha = guess.saturating_sub(window);
            let mut beta = guess.saturating_add(window);
            if alpha < -INF {
                alpha = -INF;
            }
            if beta > INF {
                beta = INF;
            }

            // Convert root-window [alpha,beta] to child-window [-beta,-alpha].
            let (child, _mv) = self.negamax(board, depth, -beta, -alpha, 1, true);

            // If we exhausted a node budget inside the search, abort and return the best guess so far.
            if self.abort {
                return guess;
            }

            let score = -child;

            // Success only when strictly inside the window.
            if score > alpha && score < beta {
                return score;
            }

            // If we've reached full window already, this must be exact.
            if alpha == -INF && beta == INF {
                return score;
            }

            // Expand and re-center on the returned bound.
            guess = score;
            if window >= INF {
                // Defensive fallback to full window.
                let (child2, _mv2) = self.negamax(board, depth, -INF, INF, 1, true);
                if self.abort {
                    return guess;
                }
                return -child2;
            }
            window = window.saturating_mul(2);
        }
    }

    /// Root exact solve for a single move using an aspiration window.
    ///
    /// Contract:
    /// - `board` is **after** applying the candidate root move.
    /// - Returned score is from the **original root side** perspective.
    fn root_score_exact_with_aspiration(
        &mut self,
        board: &mut Board,
        guess_root: Score,
        aspiration_width: Score,
    ) -> Score {
        let mut guess = guess_root;
        let mut window = aspiration_width.max(1);

        loop {
            let mut alpha = guess.saturating_sub(window);
            let mut beta = guess.saturating_add(window);
            if alpha < -INF {
                alpha = -INF;
            }
            if beta > INF {
                beta = INF;
            }

            let (child, _mv) = self.exact_negamax(board, -beta, -alpha, 1);

            // If we exhausted a node budget inside the solve, abort and return the best guess so far.
            if self.abort {
                return guess;
            }

            let score = -child;

            if score > alpha && score < beta {
                return score;
            }
            if alpha == -INF && beta == INF {
                return score;
            }

            guess = score;
            if window >= INF {
                let (child2, _mv2) = self.exact_negamax(board, -INF, INF, 1);
                if self.abort {
                    return guess;
                }
                return -child2;
            }
            window = window.saturating_mul(2);
        }
    }


    // -------------------------------------------------------------------------
    // Stage 6: Unified analyze_top_n facade (compat layer)
    // -------------------------------------------------------------------------

    /// Unified Top-N analysis entrypoint.
    ///
    /// Stage 6 introduces this façade so that *multiple* search/analysis modes can
    /// coexist behind a single stable API surface.
    ///
    /// # What this unifies
    ///
    /// The older dedicated entrypoints remain available (and are kept API-compatible),
    /// but internally they now funnel into the same shared helpers:
    ///
    /// - **Fixed** (baseline alpha-beta):
    ///   - [`Searcher::analyze_top_n_mid`]
    ///   - [`Searcher::analyze_top_n_exact`]
    /// - **Iterative** (Stage 4, Progress/Advancement):
    ///   - [`Searcher::analyze_top_n_iter`]
    /// - **Derivative** (Stage 5, best-first scheduler):
    ///   - [`crate::derivative::DerivativeEvaluator`] (`evaluatederivative` port)
    ///
    /// # Sensei mechanism mapping
    ///
    /// - `AnalyzeTopNStrategy::Fixed` / `Iterative` correspond to Sensei's
    ///   `EvaluatorAlphaBeta` family (with different outer-loop scheduling).
    /// - `AnalyzeTopNStrategy::Derivative` corresponds to Sensei's
    ///   `EvaluatorDerivative` scheduler (notably `AddChildren` + `SolvePosition`).
    ///
    /// # Parameters and defaults
    ///
    /// The four "Stage 6 defaults" are surfaced through [`AnalyzeTopNParams`]:
    /// - `seed_depth`        (Sensei: `EvaluatorThread::AddChildren` seeding depth)
    /// - `aspiration_width`  (Sensei: root aspiration window tuning)
    /// - `node_budget`       (Sensei: proof/search budget; Stage4 early stop)
    /// - `tree_node_cap`     (Sensei: fixed arena cap in `TreeNodeSupplier`)
    ///
    /// Each strategy consumes only the knobs it needs:
    /// - Fixed: uses `aspiration_width` only.
    /// - Iterative: uses `aspiration_width` + `node_budget`.
    /// - Derivative: uses `seed_depth` + `node_budget` + `tree_node_cap`.
    pub fn analyze_top_n(&mut self, board: &mut Board, req: AnalyzeTopNRequest) -> AnalyzeTopNResult {
        let top_n = req.top_n.max(1);

        match req.strategy {
            AnalyzeTopNStrategy::Fixed => {
                let pairs = match req.mode {
                    AnalyzeMode::Exact => self.analyze_top_n_exact_with_tuning(
                        board,
                        top_n,
                        req.params.aspiration_width,
                    ),
                    AnalyzeMode::Midgame { depth } => self.analyze_top_n_mid_with_tuning(
                        board,
                        depth,
                        top_n,
                        req.params.aspiration_width,
                    ),
                };

                AnalyzeTopNResult {
                    pairs,
                    stats: AnalyzeTopNStats::Fixed {
                        nodes_used: self.nodes,
                        aborted: self.abort,
                    },
                }
            }

            AnalyzeTopNStrategy::Iterative => {
                let limits = SearchLimits {
                    max_nodes: req.params.node_budget,
                };

                let res = self.analyze_top_n_iter_with_tuning(
                    board,
                    req.mode,
                    top_n,
                    limits,
                    req.stop_policy,
                    req.params.aspiration_width,
                );

                AnalyzeTopNResult {
                    pairs: res.pairs,
                    stats: AnalyzeTopNStats::Iterative(res.stats),
                }
            }

            AnalyzeTopNStrategy::Derivative => {
                let limits = SearchLimits {
                    max_nodes: req.params.node_budget,
                };

                let mut cfg = crate::derivative::DerivativeConfig::default()
                    .with_tree_node_cap(req.params.tree_node_cap);

                // Stage 6 parameter surface: `seed_depth=0` means "use DerivativeConfig defaults".
                if req.params.seed_depth != 0 {
                    cfg.seed_depth_min = req.params.seed_depth;
                    cfg.seed_depth_max = req.params.seed_depth;
                }

                // P5-1: Reuse a cached evaluator to avoid large per-call heap allocations
                // (important for Wasm/UI usage).
                let mut evaluator = match self.derivative_cache.take() {
                    Some(mut ev) => {
                        ev.reconfigure(cfg);
                        ev
                    }
                    None => crate::derivative::DerivativeEvaluator::new(cfg),
                };

                let der = evaluator.evaluate(self, &*board, limits);

                // Report the *current* root child estimates as a Top-N list.
                // This matches Sensei's notion of reporting bounds/estimates for root moves.
                let pairs = evaluator.root_top_n_estimates(top_n);

                self.derivative_cache = Some(evaluator);

                AnalyzeTopNResult {
                    pairs,
                    stats: AnalyzeTopNStats::Derivative(der),
                }
            }
        }
    }

    /// Stage 6 helper: analyze top-N moves, allowing Derivative to choose its alpha-beta backend.
    ///
    /// This is a backwards-compatible extension: callers that don't care can keep using
    /// [`Searcher::analyze_top_n`]. The extra parameter only affects the
    /// [`AnalyzeTopNStrategy::Derivative`] path.
    pub fn analyze_top_n_with_derivative_backend(
        &mut self,
        board: &mut Board,
        req: AnalyzeTopNRequest,
        derivative_backend: Option<crate::backend::BackendKind>,
    ) -> AnalyzeTopNResult {
        match req.strategy {
            AnalyzeTopNStrategy::Derivative => {
                let top_n = req.top_n.max(1);
                let limits = SearchLimits {
                    max_nodes: req.params.node_budget,
                };

                let mut cfg = crate::derivative::DerivativeConfig::default()
                    .with_tree_node_cap(req.params.tree_node_cap);

                if req.params.seed_depth != 0 {
                    cfg.seed_depth_min = req.params.seed_depth;
                    cfg.seed_depth_max = req.params.seed_depth;
                }

                // P5-1: Reuse a cached evaluator to avoid large per-call heap allocations.
                let mut evaluator = match self.derivative_cache.take() {
                    Some(mut ev) => {
                        ev.reconfigure(cfg);
                        ev
                    }
                    None => crate::derivative::DerivativeEvaluator::new(cfg),
                };

                let der = match derivative_backend.unwrap_or(crate::backend::BackendKind::Sonetto) {
                    crate::backend::BackendKind::Sonetto => evaluator.evaluate(self, &*board, limits),
                    crate::backend::BackendKind::SenseiAlphaBeta => {
                        let mut ab = crate::sensei_ab::SenseiAlphaBeta::new(
                            self.weights.clone(),
                            self.feats.clone(),
                            self.swap.clone(),
                            self.occ.clone(),
                        );
                        evaluator.evaluate_backend(&mut ab, &*board, limits)
                    }
                };

                let pairs = evaluator.root_top_n_estimates(top_n);

                self.derivative_cache = Some(evaluator);

                AnalyzeTopNResult {
                    pairs,
                    stats: AnalyzeTopNStats::Derivative(der),
                }
            }
            _ => self.analyze_top_n(board, req),
        }
    }

    /// Analyze top-N moves in **midgame** mode using the evaluation function.
    ///
    /// Returns a vector of `(move_bitpos, score)` sorted by score descending
    /// (from the current side-to-move perspective).
    ///
    /// Performance policy:
    /// - Fully searches only enough moves to maintain an accurate Top-N set.
    /// - For the remaining moves, uses a null-window search against the current
    ///   Top-N threshold; only re-searches with full window if the move can enter
    ///   the Top-N.
    pub fn analyze_top_n_mid(&mut self, board: &mut Board, depth: u8, top_n: usize) -> Vec<(Move, Score)> {
        self.analyze_top_n_mid_with_tuning(board, depth, top_n, DEFAULT_ASPIRATION_WIDTH)
    }

    /// Internal implementation for Stage 6 (exposes aspiration tuning).
    ///
    /// Sensei mapping: this is still `EvaluatorAlphaBeta` root analysis; only the
    /// aspiration half-window is parameterized.
    fn analyze_top_n_mid_with_tuning(
        &mut self,
        board: &mut Board,
        depth: u8,
        top_n: usize,
        aspiration_width: Score,
    ) -> Vec<(Move, Score)> {
        let depth = depth.max(1).min(MAX_PLY as u8);
        let top_n = top_n.max(1);

        // Stage 6 cleanup: analysis helpers should not inherit stale abort/budget state.
        self.nodes = 0;
        self.abort = false;

        // Clamp tunable knobs.
        let aspiration_width = aspiration_width.max(1);

        // Root init: ensure feature ids match the current bitboards.
        recompute_features_in_place(board, &self.occ);

        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);
        if mask == 0 {
            return Vec::new();
        }

        // Build root move list (ply=0 buffers).
        let mc = push_moves_from_mask(mask, &mut self.move_buf[0]);

        // Use TT suggestion if it is a midgame entry (depth < 128).
        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;
        if let Some(e) = self.tt.probe(board.hash) {
            if e.depth < TT_DEPTH_EXACT_BIAS {
                tt_move = e.best_move;
                tt_move2 = e.best_move2;
            }
        }

        // A) Root seed ordering (very cheap): eval + mobility (no extra search).
        //    This only affects the iteration order and therefore how fast the Top-N
        //    threshold tightens; it does NOT change fixed-depth semantics.
        let root_k1 = self.killers[0][0];
        let root_k2 = self.killers[0][1];
        for i in 0..mc {
            let mv = self.move_buf[0][i];

            let mut u = Undo::default();
            let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
            debug_assert!(ok);
            if !ok {
                self.score_buf[0][i] = -INF;
                continue;
            }

            let mut seed = self.root_seed_score_mid_after_move(board);
            if mv == tt_move {
                seed = seed.saturating_add(ROOT_TT_BONUS);
            }
            if tt_move2.is_some() && Some(mv) == tt_move2 {
                seed = seed.saturating_add(ROOT_TT2_BONUS);
            }
            if mv == root_k1 {
                seed = seed.saturating_add(ROOT_KILLER1_BONUS);
            }
            if mv == root_k2 {
                seed = seed.saturating_add(ROOT_KILLER2_BONUS);
            }

            board.undo_move_with_occ(&u, Some(&self.occ));

            self.score_buf[0][i] = seed;
        }
        sort_moves_desc(&mut self.move_buf[0], &mut self.score_buf[0], mc);

        let mut top: Vec<(Move, Score)> = Vec::with_capacity(top_n.min(mc));

        for i in 0..mc {
            let mv = self.move_buf[0][i];
            let seed_guess = self.score_buf[0][i];

            let mut u = Undo::default();
            let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
            debug_assert!(ok);
            if !ok {
                continue;
            }

            let score = if top.len() < top_n {
                // B) Root full-depth search with aspiration window.
                self.root_score_mid_with_aspiration(board, depth - 1, seed_guess, aspiration_width)
            } else {
                // Null-window test against current threshold.
                let threshold = top[top.len() - 1].1;
                let thr1 = threshold.saturating_add(1);
                let (child, _m2) = self.negamax(board, depth - 1, -thr1, -threshold, 1, false);
                let s_bound = -child;

                if s_bound > threshold {
                    // B) Only if the move can enter Top-N, do a full solve, but start
                    //    with a tight aspiration window around a reasonable guess.
                    let guess = seed_guess.max(threshold);
                    self.root_score_mid_with_aspiration(board, depth - 1, guess, aspiration_width)
                } else {
                    // Not good enough to enter Top-N.
                    board.undo_move_with_occ(&u, Some(&self.occ));
                    continue;
                }
            };

            board.undo_move_with_occ(&u, Some(&self.occ));

            insert_top_n(&mut top, mv, score, top_n);
        }

        top
    }


    /// Analyze top-N moves in **exact endgame** mode.
    ///
    /// This performs a perfect-play solve to game end (disc difference), using a
    /// separate TT depth tag (`>=128`) to avoid mixing with midgame entries.
    pub fn analyze_top_n_exact(&mut self, board: &mut Board, top_n: usize) -> Vec<(Move, Score)> {
        self.analyze_top_n_exact_with_tuning(board, top_n, DEFAULT_ASPIRATION_WIDTH)
    }

    /// Internal implementation for Stage 6 (exposes aspiration tuning).
    ///
    /// Sensei mapping: this is `EvaluatorAlphaBeta` in exact-to-end mode (disc-diff).
    fn analyze_top_n_exact_with_tuning(
        &mut self,
        board: &mut Board,
        top_n: usize,
        aspiration_width: Score,
    ) -> Vec<(Move, Score)> {
        let top_n = top_n.max(1);

        // Stage 6 cleanup: analysis helpers should not inherit stale abort/budget state.
        self.nodes = 0;
        self.abort = false;

        // Exact-to-end solve does not use evaluation features.
        // Skip feature recomputation here (saves work and avoids per-node feature maintenance).

        // Clamp tunable knobs.
        let aspiration_width = aspiration_width.max(1);

        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);
        if mask == 0 {
            return Vec::new();
        }

        let mc = push_moves_from_mask(mask, &mut self.move_buf[0]);

        // TT suggestion (exact entries only).
        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;
        if let Some(e) = self.tt.probe(board.hash) {
            if e.depth >= TT_DEPTH_EXACT_BIAS {
                tt_move = e.best_move;
                tt_move2 = e.best_move2;
            }
        }

        // A) Root seed ordering (very cheap): disc diff + mobility.
        let root_k1 = self.killers[0][0];
        let root_k2 = self.killers[0][1];
        for i in 0..mc {
            let mv = self.move_buf[0][i];

            let mut u = Undo::default();
            let ok = board.apply_move_no_features(mv, &mut u);
            debug_assert!(ok);
            if !ok {
                self.score_buf[0][i] = -INF;
                continue;
            }

            let mut seed = Self::root_seed_score_exact_after_move(board);
            if mv == tt_move {
                seed = seed.saturating_add(ROOT_TT_BONUS);
            }
            if tt_move2.is_some() && Some(mv) == tt_move2 {
                seed = seed.saturating_add(ROOT_TT2_BONUS);
            }
            if mv == root_k1 {
                seed = seed.saturating_add(ROOT_KILLER1_BONUS);
            }
            if mv == root_k2 {
                seed = seed.saturating_add(ROOT_KILLER2_BONUS);
            }

            board.undo_move_no_features(&u);

            self.score_buf[0][i] = seed;
        }
        sort_moves_desc(&mut self.move_buf[0], &mut self.score_buf[0], mc);

        let mut top: Vec<(Move, Score)> = Vec::with_capacity(top_n.min(mc));

        for i in 0..mc {
            let mv = self.move_buf[0][i];
            let seed_guess = self.score_buf[0][i];

            let mut u = Undo::default();
            let ok = board.apply_move_no_features(mv, &mut u);
            debug_assert!(ok);
            if !ok {
                continue;
            }

            let score = if top.len() < top_n {
                // B) Root full-depth exact solve with aspiration window.
                self.root_score_exact_with_aspiration(board, seed_guess, aspiration_width)
            } else {
                let threshold = top[top.len() - 1].1;
                let thr1 = threshold.saturating_add(1);
                let (child, _m2) = self.exact_negamax(board, -thr1, -threshold, 1);
                let s_bound = -child;

                if s_bound > threshold {
                    let guess = seed_guess.max(threshold);
                    self.root_score_exact_with_aspiration(board, guess, aspiration_width)
                } else {
                    board.undo_move_no_features(&u);
                    continue;
                }
            };

            board.undo_move_no_features(&u);
            insert_top_n(&mut top, mv, score, top_n);
        }

        top
    }


    // -------------------------------------------------------------------------
    // P2-4: Parallel Top-N analysis helpers (Rayon)
    // -------------------------------------------------------------------------

    /// Parallel top-N midgame analysis.
    ///
    /// This is primarily intended for **WASM internal threads** (rayon +
    /// SharedArrayBuffer), but can also be enabled on native builds.
    ///
    /// Design goals:
    /// - low-risk: never shares mutable `Searcher` state across threads
    /// - deterministic output: stable tie-break by `Move`
    /// - keeps total TT memory roughly constant by splitting TT MB across workers
    ///
    /// When Rayon is not available / not initialized (e.g. thread pool size=1),
    /// it falls back to the sequential implementation.
    #[cfg(feature = "parallel_rayon")]
    pub fn analyze_top_n_mid_parallel(&mut self, board: &mut Board, depth: u8, top_n: usize) -> Vec<(Move, Score)> {
        self.analyze_top_n_mid_parallel_with_tuning(board, depth, top_n, DEFAULT_ASPIRATION_WIDTH)
    }

    /// Parallel exact endgame top-N analysis.
    #[cfg(feature = "parallel_rayon")]
    pub fn analyze_top_n_exact_parallel(&mut self, board: &mut Board, top_n: usize) -> Vec<(Move, Score)> {
        self.analyze_top_n_exact_parallel_with_tuning(board, top_n, DEFAULT_ASPIRATION_WIDTH)
    }

    #[cfg(feature = "parallel_rayon")]
    fn analyze_top_n_mid_parallel_with_tuning(
        &mut self,
        board: &mut Board,
        depth: u8,
        top_n: usize,
        aspiration_width: Score,
    ) -> Vec<(Move, Score)> {
        let depth = depth.max(1).min(MAX_PLY as u8);
        let top_n = top_n.max(1);

        // Keep state sane and compatible with the sequential analysis helpers.
        self.nodes = 0;
        self.abort = false;

        // Clamp tunable knobs.
        let aspiration_width = aspiration_width.max(1);

        // Root init: ensure feature ids match the current bitboards.
        recompute_features_in_place(board, &self.occ);

        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);
        if mask == 0 {
            return Vec::new();
        }

        // Collect moves into a local array to avoid sharing `self.move_buf`.
        let mut root_moves: [Move; MAX_MOVES] = [PASS; MAX_MOVES];
        let mc = push_moves_from_mask(mask, &mut root_moves);
        if mc == 0 {
            return Vec::new();
        }

        // Threading isn't worth it for trivial cases.
        let avail_threads = rayon::current_num_threads();
        let t = avail_threads.max(1).min(mc);
        if t <= 1 || mc <= 1 {
            return self.analyze_top_n_mid_with_tuning(board, depth, top_n, aspiration_width);
        }

        // Use TT suggestion if it is a midgame entry (depth < 128).
        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;
        if let Some(e) = self.tt.probe(board.hash) {
            if e.depth < TT_DEPTH_EXACT_BIAS {
                tt_move = e.best_move;
                tt_move2 = e.best_move2;
            }
        }

        // A) Root seed guesses (cheap): eval + mobility (+ ordering bonuses).
        let root_k1 = self.killers[0][0];
        let root_k2 = self.killers[0][1];
        let mut seed_guess: Vec<Score> = vec![0; mc];
        for i in 0..mc {
            let mv = root_moves[i];

            let mut u = Undo::default();
            let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
            debug_assert!(ok);
            if !ok {
                seed_guess[i] = -INF;
                continue;
            }

            let mut seed = self.root_seed_score_mid_after_move(board);
            if mv == tt_move {
                seed = seed.saturating_add(ROOT_TT_BONUS);
            }
            if tt_move2.is_some() && Some(mv) == tt_move2 {
                seed = seed.saturating_add(ROOT_TT2_BONUS);
            }
            if mv == root_k1 {
                seed = seed.saturating_add(ROOT_KILLER1_BONUS);
            }
            if mv == root_k2 {
                seed = seed.saturating_add(ROOT_KILLER2_BONUS);
            }

            board.undo_move_with_occ(&u, Some(&self.occ));
            seed_guess[i] = seed;
        }

        // Base board snapshot for worker-local apply/undo.
        let base_board = board.clone();

        // Keep total TT memory roughly constant across workers.
        let worker_tt_mb: usize = (self.tt_mb / t).max(1);

        // Shared eval context clones (cheap) for worker construction.
        let weights = self.weights.clone();
        let feats = self.feats.clone();
        let swap = self.swap.clone();
        let occ = self.occ.clone();

        let child_depth = depth.saturating_sub(1);

        // B) Full evaluation for each move (parallel by root-split partition).
        // We intentionally compute ALL move scores so that the returned Top-N is
        // exact and deterministic.
        let parts: Vec<(Vec<(Move, Score)>, u64)> = (0..t)
            .into_par_iter()
            .map(|tid| {
                let mut worker = Searcher::new(
                    worker_tt_mb,
                    weights.clone(),
                    feats.clone(),
                    swap.clone(),
                    occ.clone(),
                );
                worker.abort = false;
                worker.nodes = 0;

                let mut local_board = base_board.clone();
                let mut out: Vec<(Move, Score)> = Vec::with_capacity((mc + t - 1) / t);

                for idx in (tid..mc).step_by(t) {
                    let mv = root_moves[idx];
                    let guess = seed_guess[idx];

                    let mut u = Undo::default();
                    let ok = local_board.apply_move_with_occ(mv, &mut u, Some(&worker.occ));
                    debug_assert!(ok);
                    if !ok {
                        continue;
                    }

                    let score = worker.root_score_mid_with_aspiration(
                        &mut local_board,
                        child_depth,
                        guess,
                        aspiration_width,
                    );
                    local_board.undo_move_with_occ(&u, Some(&worker.occ));

                    out.push((mv, score));
                }

                (out, worker.nodes)
            })
            .collect();

        // Merge + compute aggregate stats.
        let mut all: Vec<(Move, Score)> = Vec::with_capacity(mc);
        let mut total_nodes: u64 = 0;
        for (mut v, n) in parts {
            total_nodes = total_nodes.wrapping_add(n);
            all.append(&mut v);
        }
        self.nodes = total_nodes;

        // Deterministic ordering.
        all.sort_by(|(m1, s1), (m2, s2)| s2.cmp(s1).then_with(|| m1.cmp(m2)));
        all.truncate(top_n.min(all.len()));
        all
    }

    #[cfg(feature = "parallel_rayon")]
    fn analyze_top_n_exact_parallel_with_tuning(
        &mut self,
        board: &mut Board,
        top_n: usize,
        aspiration_width: Score,
    ) -> Vec<(Move, Score)> {
        let top_n = top_n.max(1);

        // Keep state sane and compatible with the sequential analysis helpers.
        self.nodes = 0;
        self.abort = false;

        // Root init: keep incremental pattern feature IDs in sync with the current bitboards.
        // Exact-to-end solve does not use evaluation features.
        // Skip feature recomputation here (saves work and avoids per-node feature maintenance).

        // Clamp tunable knobs.
        let aspiration_width = aspiration_width.max(1);

        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);
        if mask == 0 {
            return Vec::new();
        }

        // Collect moves into a local array to avoid sharing `self.move_buf`.
        let mut root_moves: [Move; MAX_MOVES] = [PASS; MAX_MOVES];
        let mc = push_moves_from_mask(mask, &mut root_moves);
        if mc == 0 {
            return Vec::new();
        }

        let avail_threads = rayon::current_num_threads();
        let t = avail_threads.max(1).min(mc);
        if t <= 1 || mc <= 1 {
            return self.analyze_top_n_exact_with_tuning(board, top_n, aspiration_width);
        }

        // TT suggestion (exact entries only).
        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;
        if let Some(e) = self.tt.probe(board.hash) {
            if e.depth >= TT_DEPTH_EXACT_BIAS {
                tt_move = e.best_move;
                tt_move2 = e.best_move2;
            }
        }

        // A) Root seed guesses (cheap): disc diff + mobility (+ ordering bonuses).
        let root_k1 = self.killers[0][0];
        let root_k2 = self.killers[0][1];
        let mut seed_guess: Vec<Score> = vec![0; mc];
        for i in 0..mc {
            let mv = root_moves[i];

            let mut u = Undo::default();
            let ok = board.apply_move_no_features(mv, &mut u);
            debug_assert!(ok);
            if !ok {
                seed_guess[i] = -INF;
                continue;
            }

            let mut seed = Self::root_seed_score_exact_after_move(board);
            if mv == tt_move {
                seed = seed.saturating_add(ROOT_TT_BONUS);
            }
            if tt_move2.is_some() && Some(mv) == tt_move2 {
                seed = seed.saturating_add(ROOT_TT2_BONUS);
            }
            if mv == root_k1 {
                seed = seed.saturating_add(ROOT_KILLER1_BONUS);
            }
            if mv == root_k2 {
                seed = seed.saturating_add(ROOT_KILLER2_BONUS);
            }

            board.undo_move_no_features(&u);
            seed_guess[i] = seed;
        }

        // Base board snapshot for worker-local apply/undo.
        let base_board = board.clone();

        // Keep total TT memory roughly constant across workers.
        let worker_tt_mb: usize = (self.tt_mb / t).max(1);

        // Shared eval context clones (cheap) for worker construction.
        let weights = self.weights.clone();
        let feats = self.feats.clone();
        let swap = self.swap.clone();
        let occ = self.occ.clone();

        // B) Full evaluation for each move (parallel by root-split partition).
        let parts: Vec<(Vec<(Move, Score)>, u64)> = (0..t)
            .into_par_iter()
            .map(|tid| {
                let mut worker = Searcher::new(
                    worker_tt_mb,
                    weights.clone(),
                    feats.clone(),
                    swap.clone(),
                    occ.clone(),
                );
                worker.abort = false;
                worker.nodes = 0;

                let mut local_board = base_board.clone();
                let mut out: Vec<(Move, Score)> = Vec::with_capacity((mc + t - 1) / t);

                for idx in (tid..mc).step_by(t) {
                    let mv = root_moves[idx];
                    let guess = seed_guess[idx];

                    let mut u = Undo::default();
                    let ok = local_board.apply_move_no_features(mv, &mut u);
                    debug_assert!(ok);
                    if !ok {
                        continue;
                    }

                    let score = worker.root_score_exact_with_aspiration(
                        &mut local_board,
                        guess,
                        aspiration_width,
                    );
                    local_board.undo_move_no_features(&u);

                    out.push((mv, score));
                }

                (out, worker.nodes)
            })
            .collect();

        let mut all: Vec<(Move, Score)> = Vec::with_capacity(mc);
        let mut total_nodes: u64 = 0;
        for (mut v, n) in parts {
            total_nodes = total_nodes.wrapping_add(n);
            all.append(&mut v);
        }
        self.nodes = total_nodes;

        all.sort_by(|(m1, s1), (m2, s2)| s2.cmp(s1).then_with(|| m1.cmp(m2)));
        all.truncate(top_n.min(all.len()));
        all
    }



    /// Parallel iterative Top-N analysis (root-split).
    ///
    /// This helper exists primarily for WASM threads (wasm-bindgen-rayon): we
    /// distribute expensive root move searches across workers while keeping the
    /// total TT memory roughly constant.
    ///
    /// Note:
    /// - This is a pragmatic throughput path for UI analysis. It does *not* try
    ///   to reproduce Stage4's full iterative refinement logic exactly.
    /// - When a node budget is provided, it is split across workers. If a worker
    ///   exhausts its budget, remaining assigned candidate moves fall back to
    ///   their cheap seed guess.
    #[cfg(feature = "parallel_rayon")]
    pub fn analyze_top_n_iter_parallel(
        &mut self,
        board: &mut Board,
        mode: AnalyzeMode,
        top_n: usize,
        limits: SearchLimits,
        stop_policy: StopPolicy,
        aspiration_width: Score,
    ) -> AnalyzeIterResult {
        let top_n = top_n.max(1);

        // Keep state sane and compatible with the sequential analysis helpers.
        self.nodes = 0;
        self.abort = false;

        // Clamp tunable knobs.
        let aspiration_width = aspiration_width.max(1);

        // Root init: keep incremental pattern feature IDs in sync with the current bitboards.
        // Exact-to-end solve does not use evaluation features.
        let (is_exact, child_depth) = match mode {
            AnalyzeMode::Exact => (true, 0u8),
            AnalyzeMode::Midgame { depth } => {
                // Clamp to buffer depth to avoid any OOB on adversarial inputs.
                let depth = depth.max(1).min(MAX_PLY as u8);
                recompute_features_in_place(board, &self.occ);
                (false, depth.saturating_sub(1))
            }
        };

        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);
        if mask == 0 {
            return AnalyzeIterResult {
                pairs: Vec::new(),
                stats: AnalyzeIterStats {
                    nodes_used: 0,
                    rounds: 0,
                    gap: 0,
                    complete: true,
                    stop_reason: StopReason::NoMoves,
                },
            };
        }

        // Collect moves into a local array to avoid sharing `self.move_buf`.
        let mut root_moves: [Move; MAX_MOVES] = [PASS; MAX_MOVES];
        let mc = push_moves_from_mask(mask, &mut root_moves);
        if mc == 0 {
            return AnalyzeIterResult {
                pairs: Vec::new(),
                stats: AnalyzeIterStats {
                    nodes_used: 0,
                    rounds: 0,
                    gap: 0,
                    complete: true,
                    stop_reason: StopReason::NoMoves,
                },
            };
        }
        let top_n = top_n.min(mc).max(1);

        let avail_threads = rayon::current_num_threads();
        let t = avail_threads.max(1).min(mc);
        if t <= 1 || mc <= 1 {
            return self.analyze_top_n_iter_with_tuning(
                board,
                mode,
                top_n,
                limits,
                stop_policy,
                aspiration_width,
            );
        }

        // TT suggestion (use the entry class matching the analyze mode).
        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;
        if let Some(e) = self.tt.probe(board.hash) {
            if (is_exact && e.depth >= TT_DEPTH_EXACT_BIAS)
                || (!is_exact && e.depth < TT_DEPTH_EXACT_BIAS)
            {
                tt_move = e.best_move;
                tt_move2 = e.best_move2;
            }
        }

        // A) Root seed guesses (cheap): eval + mobility (+ ordering bonuses).
        let root_k1 = self.killers[0][0];
        let root_k2 = self.killers[0][1];
        let mut seed_guess: Vec<Score> = vec![0; mc];
        for i in 0..mc {
            let mv = root_moves[i];

            let mut seed: Score;
            if is_exact {
                let mut u = Undo::default();
                let ok = board.apply_move_no_features(mv, &mut u);
                debug_assert!(ok);
                if !ok {
                    seed_guess[i] = -INF;
                    continue;
                }

                seed = Self::root_seed_score_exact_after_move(board);
                if mv == tt_move {
                    seed = seed.saturating_add(ROOT_TT_BONUS);
                }
                if tt_move2.is_some() && Some(mv) == tt_move2 {
                    seed = seed.saturating_add(ROOT_TT2_BONUS);
                }
                if mv == root_k1 {
                    seed = seed.saturating_add(ROOT_KILLER1_BONUS);
                }
                if mv == root_k2 {
                    seed = seed.saturating_add(ROOT_KILLER2_BONUS);
                }

                board.undo_move_no_features(&u);
            } else {
                let mut u = Undo::default();
                let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
                debug_assert!(ok);
                if !ok {
                    seed_guess[i] = -INF;
                    continue;
                }

                seed = self.root_seed_score_mid_after_move(board);
                if mv == tt_move {
                    seed = seed.saturating_add(ROOT_TT_BONUS);
                }
                if tt_move2.is_some() && Some(mv) == tt_move2 {
                    seed = seed.saturating_add(ROOT_TT2_BONUS);
                }
                if mv == root_k1 {
                    seed = seed.saturating_add(ROOT_KILLER1_BONUS);
                }
                if mv == root_k2 {
                    seed = seed.saturating_add(ROOT_KILLER2_BONUS);
                }

                board.undo_move_with_occ(&u, Some(&self.occ));
            }

            seed_guess[i] = seed;
        }

        // Candidate set: for GoodStop / budgeted analysis, focus on a handful of
        // promising moves (still returning a full Top-N among *all* legal moves).
        let mut k = mc;
        if stop_policy == StopPolicy::GoodStop || limits.max_nodes.is_some() {
            k = (top_n * 4).max(t).min(mc);
        }

        let mut idxs: Vec<usize> = (0..mc).collect();
        idxs.sort_by(|&a, &b| {
            seed_guess[b]
                .cmp(&seed_guess[a])
                .then_with(|| root_moves[a].cmp(&root_moves[b]))
        });
        let mut cand_moves: Vec<Move> = Vec::with_capacity(k);
        let mut cand_guess: Vec<Score> = Vec::with_capacity(k);
        for &i in idxs.iter().take(k) {
            cand_moves.push(root_moves[i]);
            cand_guess.push(seed_guess[i]);
        }

        // Base board snapshot for worker-local apply/undo.
        let base_board = board.clone();

        // Keep total TT memory roughly constant across workers.
        let worker_tt_mb: usize = (self.tt_mb / t).max(1);

        // Split the node budget across workers to preserve the meaning of
        // `limits.max_nodes` as an overall budget for the analysis call.
        let worker_limits: SearchLimits = match limits.max_nodes {
            Some(n) => SearchLimits {
                max_nodes: Some((n / t as u64).max(1)),
            },
            None => limits,
        };

        // Shared eval context clones (cheap) for worker construction.
        let weights = self.weights.clone();
        let feats = self.feats.clone();
        let swap = self.swap.clone();
        let occ = self.occ.clone();

        // B) Full evaluation for the candidate set (parallel by root-split partition).
        let parts: Vec<(Vec<(Move, Score)>, u64, bool)> = (0..t)
            .into_par_iter()
            .map(|tid| {
                let mut worker = Searcher::new(
                    worker_tt_mb,
                    weights.clone(),
                    feats.clone(),
                    swap.clone(),
                    occ.clone(),
                );
                worker.limits = worker_limits;
                worker.abort = false;
                worker.nodes = 0;

                let mut local_board = base_board.clone();
                let mut out: Vec<(Move, Score)> = Vec::with_capacity((k + t - 1) / t);

                for idx in (tid..k).step_by(t) {
                    let mv = cand_moves[idx];
                    let guess = cand_guess[idx];

                    // If this worker already hit its node budget, fall back to the seed.
                    if worker.abort {
                        out.push((mv, guess));
                        continue;
                    }

                    let mut u = Undo::default();
                    let ok = if is_exact {
                        local_board.apply_move_no_features(mv, &mut u)
                    } else {
                        local_board.apply_move_with_occ(mv, &mut u, Some(&worker.occ))
                    };
                    debug_assert!(ok);
                    if !ok {
                        out.push((mv, -INF));
                        continue;
                    }

                    let score = if is_exact {
                        worker.root_score_exact_with_aspiration(
                            &mut local_board,
                            guess,
                            aspiration_width,
                        )
                    } else {
                        worker.root_score_mid_with_aspiration(
                            &mut local_board,
                            child_depth,
                            guess,
                            aspiration_width,
                        )
                    };

                    if is_exact {
                        local_board.undo_move_no_features(&u);
                    } else {
                        local_board.undo_move_with_occ(&u, Some(&worker.occ));
                    }

                    out.push((mv, score));
                }

                (out, worker.nodes, worker.abort)
            })
            .collect();

        // Build a per-move score table (default: seed guess), then patch in the
        // searched candidate results.
        let mut score_by_mv: [Score; 64] = [0; 64];
        for i in 0..mc {
            let mv = root_moves[i];
            score_by_mv[mv as usize] = seed_guess[i];
        }

        let mut total_nodes: u64 = 0;
        let mut aborted_any = false;
        for (v, n, ab) in parts {
            total_nodes = total_nodes.wrapping_add(n);
            aborted_any |= ab;
            for (mv, sc) in v {
                score_by_mv[mv as usize] = sc;
            }
        }

        self.nodes = total_nodes;
        self.abort = aborted_any;

        let mut all: Vec<(Move, Score)> = Vec::with_capacity(mc);
        for i in 0..mc {
            let mv = root_moves[i];
            all.push((mv, score_by_mv[mv as usize]));
        }

        all.sort_by(|(m1, s1), (m2, s2)| s2.cmp(s1).then_with(|| m1.cmp(m2)));
        all.truncate(top_n.min(all.len()));

        AnalyzeIterResult {
            pairs: all,
            stats: AnalyzeIterStats {
                nodes_used: total_nodes,
                rounds: 1,
                gap: 0,
                complete: !aborted_any,
                stop_reason: if aborted_any {
                    StopReason::Budget
                } else {
                    StopReason::Complete
                },
            },
        }
    }

    /// Parallel derivative Top-N analysis (root-split).
    ///
    /// We run the derivative scheduler independently for a small candidate set
    /// of promising root moves (selected by a cheap seed score), in parallel.
    /// Remaining moves keep their seed score.
    #[cfg(feature = "parallel_rayon")]
    pub fn analyze_top_n_derivative_parallel(
        &mut self,
        board: &mut Board,
        top_n: usize,
        limits: SearchLimits,
        tree_node_cap: usize,
        seed_depth: u8,
    ) -> Vec<(Move, Score)> {
        let top_n = top_n.max(1);

        // Keep state sane and compatible with the sequential analysis helpers.
        self.nodes = 0;
        self.abort = false;

        // Derivative mode uses the evaluation feature set during expansion.
        recompute_features_in_place(board, &self.occ);

        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);
        if mask == 0 {
            return Vec::new();
        }

        // Collect moves into a local array to avoid sharing `self.move_buf`.
        let mut root_moves: [Move; MAX_MOVES] = [PASS; MAX_MOVES];
        let mc = push_moves_from_mask(mask, &mut root_moves);
        if mc == 0 {
            return Vec::new();
        }
        let top_n = top_n.min(mc).max(1);

        let avail_threads = rayon::current_num_threads();
        let t = avail_threads.max(1).min(mc);
        if t <= 1 || mc <= 1 {
            // Fall back to the shared single-thread derivative scheduler.
            let mut cfg = crate::derivative::DerivativeConfig::default().with_tree_node_cap(tree_node_cap);
            if seed_depth > 0 {
                cfg.seed_depth_min = seed_depth;
                cfg.seed_depth_max = seed_depth;
            }
            let mut evaluator = crate::derivative::DerivativeEvaluator::new(cfg);
            let der = evaluator.evaluate(self, &*board, limits);
            self.nodes = der.nodes_used;
            self.abort = matches!(der.status, crate::derivative::DerivativeStatus::Budget);
            return evaluator.root_top_n_estimates(top_n);
        }

        // A) Root seed guesses (cheap): eval + mobility (+ ordering bonuses).
        // Use midgame TT entries only.
        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;
        if let Some(e) = self.tt.probe(board.hash) {
            if e.depth < TT_DEPTH_EXACT_BIAS {
                tt_move = e.best_move;
                tt_move2 = e.best_move2;
            }
        }

        let root_k1 = self.killers[0][0];
        let root_k2 = self.killers[0][1];
        let mut seed_guess: Vec<Score> = vec![0; mc];
        for i in 0..mc {
            let mv = root_moves[i];

            let mut u = Undo::default();
            let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
            debug_assert!(ok);
            if !ok {
                seed_guess[i] = -INF;
                continue;
            }

            let mut seed = self.root_seed_score_mid_after_move(board);
            if mv == tt_move {
                seed = seed.saturating_add(ROOT_TT_BONUS);
            }
            if tt_move2.is_some() && Some(mv) == tt_move2 {
                seed = seed.saturating_add(ROOT_TT2_BONUS);
            }
            if mv == root_k1 {
                seed = seed.saturating_add(ROOT_KILLER1_BONUS);
            }
            if mv == root_k2 {
                seed = seed.saturating_add(ROOT_KILLER2_BONUS);
            }

            board.undo_move_with_occ(&u, Some(&self.occ));
            seed_guess[i] = seed;
        }

        // Candidate set: keep it small (derivative runs are heavier than a single
        // fixed-depth root search).
        let k = (top_n * 4).max(t).min(mc);

        let mut idxs: Vec<usize> = (0..mc).collect();
        idxs.sort_by(|&a, &b| {
            seed_guess[b]
                .cmp(&seed_guess[a])
                .then_with(|| root_moves[a].cmp(&root_moves[b]))
        });
        let mut cand_moves: Vec<Move> = Vec::with_capacity(k);
        for &i in idxs.iter().take(k) {
            cand_moves.push(root_moves[i]);
        }

        // Split the global node budget evenly across evaluated candidates.
        let per_limits: SearchLimits = match limits.max_nodes {
            Some(n) => SearchLimits {
                max_nodes: Some((n / k as u64).max(1)),
            },
            None => limits,
        };

        let mut cfg = crate::derivative::DerivativeConfig::default().with_tree_node_cap(tree_node_cap);
        if seed_depth > 0 {
            cfg.seed_depth_min = seed_depth;
            cfg.seed_depth_max = seed_depth;
        }

        let base_board = board.clone();

        // Keep total TT memory roughly constant across workers.
        let worker_tt_mb: usize = (self.tt_mb / t).max(1);

        let weights = self.weights.clone();
        let feats = self.feats.clone();
        let swap = self.swap.clone();
        let occ = self.occ.clone();

        let parts: Vec<(Vec<(Move, Score)>, u64, bool)> = (0..t)
            .into_par_iter()
            .map(|tid| {
                let mut worker = Searcher::new(
                    worker_tt_mb,
                    weights.clone(),
                    feats.clone(),
                    swap.clone(),
                    occ.clone(),
                );
                worker.abort = false;
                worker.nodes = 0;

                let mut evaluator = crate::derivative::DerivativeEvaluator::new(cfg);

                let mut local_board = base_board.clone();
                let mut out: Vec<(Move, Score)> = Vec::with_capacity((k + t - 1) / t);
                let mut nodes_used: u64 = 0;
                let mut budgeted = false;

                for idx in (tid..k).step_by(t) {
                    let mv = cand_moves[idx];

                    let mut u = Undo::default();
                    let ok = local_board.apply_move_with_occ(mv, &mut u, Some(&worker.occ));
                    debug_assert!(ok);
                    if !ok {
                        out.push((mv, -INF));
                        continue;
                    }

                    let der = evaluator.evaluate(&mut worker, &local_board, per_limits);
                    nodes_used = nodes_used.wrapping_add(der.nodes_used);
                    budgeted |= matches!(der.status, crate::derivative::DerivativeStatus::Budget);

                    // Convert child-to-move score to the root side.
                    let score = der.estimate.saturating_neg();

                    local_board.undo_move_with_occ(&u, Some(&worker.occ));
                    out.push((mv, score));
                }

                (out, nodes_used, budgeted)
            })
            .collect();

        // Build a per-move score table (default: seed guess), then patch in the
        // derivative candidate results.
        let mut score_by_mv: [Score; 64] = [0; 64];
        for i in 0..mc {
            let mv = root_moves[i];
            score_by_mv[mv as usize] = seed_guess[i];
        }

        let mut total_nodes: u64 = 0;
        let mut budgeted_any = false;
        for (v, n, b) in parts {
            total_nodes = total_nodes.wrapping_add(n);
            budgeted_any |= b;
            for (mv, sc) in v {
                score_by_mv[mv as usize] = sc;
            }
        }

        self.nodes = total_nodes;
        self.abort = budgeted_any;

        let mut all: Vec<(Move, Score)> = Vec::with_capacity(mc);
        for i in 0..mc {
            let mv = root_moves[i];
            all.push((mv, score_by_mv[mv as usize]));
        }

        all.sort_by(|(m1, s1), (m2, s2)| s2.cmp(s1).then_with(|| m1.cmp(m2)));
        all.truncate(top_n.min(all.len()));
        all
    }

    // -------------------------------------------------------------------------
    // Stage 4: Iterative Top-N analysis with budget + progress (Derivative-style)
    // -------------------------------------------------------------------------

    /// Iterative Top-N analysis that can converge to the same result as a fixed-depth
    /// analysis, while supporting:
    /// - per-move state management (Progress/Advancement)
    /// - node-budget early stop
    /// - "good_stop" heuristic based on boundary-gap shrink rate
    ///
    /// Semantics:
    /// - When `stop_policy` is [`StopPolicy::Complete`] **and** the node budget is not
    ///   exhausted, the returned `pairs` are equivalent to `analyze_top_n_mid` (midgame)
    ///   or `analyze_top_n_exact` (exact) with the same `top_n`.
    /// - Move elimination (marking a move as unable to enter the Top-N) is only performed
    ///   after a target-depth null-window verification.
    pub fn analyze_top_n_iter(
        &mut self,
        board: &mut Board,
        mode: AnalyzeMode,
        top_n: usize,
        limits: SearchLimits,
        stop_policy: StopPolicy,
    ) -> AnalyzeIterResult {
        self.analyze_top_n_iter_with_tuning(
            board,
            mode,
            top_n,
            limits,
            stop_policy,
            DEFAULT_ASPIRATION_WIDTH,
        )
    }

    /// Internal implementation for Stage 6 (exposes aspiration tuning).
    fn analyze_top_n_iter_with_tuning(
        &mut self,
        board: &mut Board,
        mode: AnalyzeMode,
        top_n: usize,
        limits: SearchLimits,
        stop_policy: StopPolicy,
        aspiration_width: Score,
    ) -> AnalyzeIterResult {
        // --- config / clamps ---
        let top_n = top_n.max(1);

        // Stage 6: configurable aspiration window width for root re-searches.
        // (Sensei: aspiration window tuning inside EvaluatorAlphaBeta)
        let aspiration_width = aspiration_width.max(1);

        let (is_exact, child_depth): (bool, u8) = match mode {
            AnalyzeMode::Exact => (true, 0),
            AnalyzeMode::Midgame { depth } => {
                let d = depth.max(1).min(MAX_PLY as u8);
                (false, d - 1)
            }
        };

        // Root init: if the caller allocated incremental feature storage, ensure it matches the
        // current bitboards before we start applying incremental updates.
        recompute_features_in_place(board, &self.occ);

        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);
        if mask == 0 {
            return AnalyzeIterResult {
                pairs: Vec::new(),
                stats: AnalyzeIterStats {
                    nodes_used: 0,
                    rounds: 0,
                    gap: 0,
                    complete: true,
                    stop_reason: StopReason::NoMoves,
                },
            };
        }

        // Build root move list (ply=0 buffers).
        let mc = push_moves_from_mask(mask, &mut self.move_buf[0]);
        let top_n = top_n.min(mc).max(1);

        // TT suggestion depending on mode.
        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;
        if let Some(e) = self.tt.probe(board.hash) {
            if is_exact {
                if e.depth >= TT_DEPTH_EXACT_BIAS {
                    tt_move = e.best_move;
                    tt_move2 = e.best_move2;
                }
            } else {
                if e.depth < TT_DEPTH_EXACT_BIAS {
                    tt_move = e.best_move;
                    tt_move2 = e.best_move2;
                }
            }
        }

        // Root killer suggestions.
        let root_k1 = self.killers[0][0];
        let root_k2 = self.killers[0][1];

        // Seed ordering (cheap, does not affect fixed-depth semantics).
        for i in 0..mc {
            let mv = self.move_buf[0][i];

            let mut u = Undo::default();
            let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
            debug_assert!(ok);
            if !ok {
                self.score_buf[0][i] = -INF;
                continue;
            }

            let mut seed = if is_exact {
                Self::root_seed_score_exact_after_move(board)
            } else {
                self.root_seed_score_mid_after_move(board)
            };

            if mv == tt_move {
                seed = seed.saturating_add(ROOT_TT_BONUS);
            }
            if tt_move2.is_some() && Some(mv) == tt_move2 {
                seed = seed.saturating_add(ROOT_TT2_BONUS);
            }
            if mv == root_k1 {
                seed = seed.saturating_add(ROOT_KILLER1_BONUS);
            }
            if mv == root_k2 {
                seed = seed.saturating_add(ROOT_KILLER2_BONUS);
            }

            board.undo_move_with_occ(&u, Some(&self.occ));

            self.score_buf[0][i] = seed;
        }
        sort_moves_desc(&mut self.move_buf[0], &mut self.score_buf[0], mc);

        // Per-move state (Stage4 requirement A).
        #[derive(Clone, Copy, Debug)]
        struct MoveState {
            mv: Move,
            seed: Score,
            lo: Score,
            hi: Score,
            solved: bool,
            eliminated: bool,
        }

        let mut states: Vec<MoveState> = Vec::with_capacity(mc);
        for i in 0..mc {
            let mv = self.move_buf[0][i];
            // `score_buf` at ply 0 contains the root seed used for ordering and as an aspiration hint.
            let seed = self.score_buf[0][i] as Score;

            // Safe initial bounds.
            let mut lo = SCORE_MIN;
            let mut hi = SCORE_MAX;

            // In exact mode we can cheaply tighten bounds using stability.
            #[cfg(feature = "stability_cutoff")]
            {
                if is_exact {
                    let mut u = Undo::default();
                    let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
                    debug_assert!(ok);
                    if ok {
                        let (c_lo, c_hi) = stability_bounds_for_side_to_move(board);
                        // Child is opponent to move; root value is negated.
                        lo = (-c_hi).clamp(SCORE_MIN, SCORE_MAX);
                        hi = (-c_lo).clamp(SCORE_MIN, SCORE_MAX);
                        board.undo_move_with_occ(&u, Some(&self.occ));
                    }
                }
            }

            states.push(MoveState {
                mv,
                seed,
                lo,
                hi,
                solved: false,
                eliminated: false,
            });
        }

        // Apply limits for this call only.
        let old_limits = self.limits;
        let old_abort = self.abort;
        self.limits = limits;
        self.abort = false;
        self.nodes = 0;

        // Top-N list of solved moves (exact scores for those moves).
        let mut top: Vec<(Move, Score)> = Vec::with_capacity(top_n.min(mc));

        // Progress tracking (Stage4 requirement C).
        let mut rounds: u32 = 0;
        let mut prev_gap: Score = 0;
        let mut prev_nodes: u64 = 0;
        let mut stall_rounds: u32 = 0;

        // "good_stop" tuning knobs (heuristic; conservative defaults).
        const GOOD_STOP_MIN_NODES: u64 = 25_000;
        const GOOD_STOP_STALL_ITERS: u32 = 3;
        // Minimum acceptable gap shrink per 100k nodes (Score units).
        const GOOD_STOP_MIN_EFF_PER_100K: i64 = 1;
        // Also allow stop when gap is already this small (Score units).
        const GOOD_STOP_GAP_TARGET: Score = 2 * SCALE; // ~2 discs

        // Hard safety cap: ensures termination even under pathological conditions.
        let max_rounds: u32 = (mc as u32).saturating_mul(8).max(1);

        let mut stop_reason = StopReason::MaxRounds;

        // Helper to compute the current boundary gap:
        // gap = max(0, max_outside_hi - threshold).
        let compute_gap = |states: &Vec<MoveState>, threshold: Score| -> Score {
            let mut max_hi = SCORE_MIN;
            for s in states.iter() {
                if s.solved || s.eliminated {
                    continue;
                }
                if s.hi > max_hi {
                    max_hi = s.hi;
                }
            }
            if max_hi == SCORE_MIN {
                0
            } else {
                (max_hi.saturating_sub(threshold)).max(0)
            }
        };

        // Main iterative loop: each round advances exactly one move.
        loop {
            // Budget stop.
            if self.abort {
                stop_reason = StopReason::Budget;
                break;
            }
            if let Some(max) = self.limits.max_nodes {
                if self.nodes >= max {
                    stop_reason = StopReason::Budget;
                    break;
                }
            }
            if rounds >= max_rounds {
                // stop_reason defaults to MaxRounds.
                break;
            }

            // Completion check (StopPolicy::Complete): all moves processed.
            let processed_all = states.iter().all(|s| s.solved || s.eliminated);
            if processed_all && top.len() >= top_n {
                stop_reason = StopReason::Complete;
                break;
            }

            // Ensure we have at least `top_n` solved moves to define a threshold.
            if top.len() < top_n {
                // Pick the move with the highest potential upper bound (tie-break by seed).
                let mut best_idx: Option<usize> = None;
                let mut best_hi: Score = SCORE_MIN;
                let mut best_seed: Score = SCORE_MIN;
                for (idx, s) in states.iter().enumerate() {
                    if s.solved || s.eliminated {
                        continue;
                    }
                    if s.hi > best_hi || (s.hi == best_hi && s.seed > best_seed) {
                        best_hi = s.hi;
                        best_seed = s.seed;
                        best_idx = Some(idx);
                    }
                }

                if let Some(idx) = best_idx {
                    let mv = states[idx].mv;
                    let guess = states[idx].seed;

                    let mut u = Undo::default();
                    let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
                    if !ok {
                        states[idx].eliminated = true;
                        rounds = rounds.saturating_add(1);
                        continue;
                    }

                    let score = if is_exact {
                        self.root_score_exact_with_aspiration(board, guess, aspiration_width)
                    } else {
                        self.root_score_mid_with_aspiration(board, child_depth, guess, aspiration_width)
                    };

                    board.undo_move_with_occ(&u, Some(&self.occ));

                    if self.abort {
                        stop_reason = StopReason::Budget;
                        break;
                    }

                    states[idx].solved = true;
                    states[idx].lo = score;
                    states[idx].hi = score;
                    insert_top_n(&mut top, mv, score, top_n);

                    // Initialize gap tracking once threshold exists.
                    if top.len() == top_n {
                        let threshold = top[top.len() - 1].1;
                        prev_gap = compute_gap(&states, threshold);
                        prev_nodes = self.nodes;
                    }

                    rounds = rounds.saturating_add(1);
                    continue;
                } else {
                    stop_reason = StopReason::NoMoves;
                    break;
                }
            }

            // We have a threshold.
            let threshold = top[top.len() - 1].1;
            let gap_before = compute_gap(&states, threshold);

            // Optional good_stop: if bounds already separate the Top-N, we can stop.
            if stop_policy == StopPolicy::GoodStop {
                if gap_before == 0 {
                    stop_reason = StopReason::GoodStop;
                    break;
                }
                if gap_before <= GOOD_STOP_GAP_TARGET && self.nodes >= GOOD_STOP_MIN_NODES {
                    stop_reason = StopReason::GoodStop;
                    break;
                }
            }

            // Pick the most critical unprocessed move: maximal (hi - threshold).
            let mut best_idx: Option<usize> = None;
            let mut best_crit: Score = SCORE_MIN;
            let mut best_seed: Score = SCORE_MIN;
            for (idx, s) in states.iter().enumerate() {
                if s.solved || s.eliminated {
                    continue;
                }
                let crit = s.hi.saturating_sub(threshold);
                if crit > best_crit || (crit == best_crit && s.seed > best_seed) {
                    best_crit = crit;
                    best_seed = s.seed;
                    best_idx = Some(idx);
                }
            }

            let Some(idx) = best_idx else {
                // Nothing left to process.
                stop_reason = StopReason::Complete;
                break;
            };

            let mv = states[idx].mv;

            // ------------------------------
            // B) Lossless verification gate
            // ------------------------------
            // Before discarding a move, verify at target depth with a null-window.
            let mut u = Undo::default();
            let ok = board.apply_move_with_occ(mv, &mut u, Some(&self.occ));
            if !ok {
                states[idx].eliminated = true;
                rounds = rounds.saturating_add(1);
                continue;
            }

            let thr = threshold.clamp(-INF, INF);
            let thr1 = thr.saturating_add(1);

            let can_beat = if is_exact {
                let (child, _m2) = self.exact_negamax(board, -thr1, -thr, 1);
                if self.abort {
                    board.undo_move_with_occ(&u, Some(&self.occ));
                    stop_reason = StopReason::Budget;
                    break;
                }
                let s_bound = -child;
                s_bound > thr
            } else {
                let (child, _m2) = self.negamax(board, child_depth, -thr1, -thr, 1, false);
                if self.abort {
                    board.undo_move_with_occ(&u, Some(&self.occ));
                    stop_reason = StopReason::Budget;
                    break;
                }
                let s_bound = -child;
                s_bound > thr
            };

            if !can_beat {
                // Verified eliminated.
                states[idx].eliminated = true;
                states[idx].hi = thr;

                board.undo_move_with_occ(&u, Some(&self.occ));
            } else {
                // Must solve to know whether it enters the Top-N.
                let guess = states[idx].seed.max(threshold);
                let score = if is_exact {
                    self.root_score_exact_with_aspiration(board, guess, aspiration_width)
                } else {
                    self.root_score_mid_with_aspiration(board, child_depth, guess, aspiration_width)
                };

                board.undo_move_with_occ(&u, Some(&self.occ));

                if self.abort {
                    stop_reason = StopReason::Budget;
                    break;
                }

                states[idx].solved = true;
                states[idx].lo = score;
                states[idx].hi = score;
                insert_top_n(&mut top, mv, score, top_n);
            }

            rounds = rounds.saturating_add(1);

            // Update progress metrics for good_stop.
            if stop_policy == StopPolicy::GoodStop {
                let new_threshold = top[top.len() - 1].1;
                let gap_after = compute_gap(&states, new_threshold);

                let nodes_delta = self.nodes.saturating_sub(prev_nodes) as i64;
                let shrink = prev_gap.saturating_sub(gap_after) as i64;

                if nodes_delta > 0 {
                    let eff_per_100k = (shrink * 100_000) / nodes_delta;
                    if eff_per_100k < GOOD_STOP_MIN_EFF_PER_100K {
                        stall_rounds = stall_rounds.saturating_add(1);
                    } else {
                        stall_rounds = 0;
                    }
                }

                prev_gap = gap_after;
                prev_nodes = self.nodes;

                if self.nodes >= GOOD_STOP_MIN_NODES && stall_rounds >= GOOD_STOP_STALL_ITERS {
                    stop_reason = StopReason::GoodStop;
                    break;
                }
            }
        }

        // Compute final gap for reporting.
        let final_gap = if top.len() >= top_n {
            let thr = top[top.len() - 1].1;
            compute_gap(&states, thr)
        } else {
            0
        };

        let complete = stop_reason == StopReason::Complete;

        let result = AnalyzeIterResult {
            pairs: top,
            stats: AnalyzeIterStats {
                nodes_used: self.nodes,
                rounds,
                gap: final_gap,
                complete,
                stop_reason,
            },
        };

        // Restore caller state.
        self.limits = old_limits;
        self.abort = old_abort;

        result
    
    }

    /// Exact endgame negamax (to completion), alpha-beta with TT.
    ///
    /// Depth is implicitly `empty_count` (passes do not consume empties).
    #[inline(always)]
    fn apply_endgame_parity_ordering(&mut self, ply: usize, mc: usize, parity: u8, keep_front: Move) {
        debug_assert!(mc <= MAX_MOVES);
        if mc <= 1 || parity == 0 {
            return;
        }

        let flips_valid = self.flips_valid[ply];
        let moves = &mut self.move_buf[ply];

        // Stable partition: parity-favoured moves first.
        //
        // P0-9: avoid per-node `[PASS; MAX_MOVES]` / `[0; MAX_MOVES]` initialization by
        // reusing scratch buffers stored on the searcher.
        if flips_valid {
            // P1-1: keep the cached flips aligned with `moves`.
            let flips = &mut self.flips_buf[ply];
            let tmp_moves = &mut self.parity_tmp_moves;
            let tmp_flips = &mut self.parity_tmp_flips;
            let mut n = 0usize;

            for i in 0..mc {
                let mv = moves[i];
                if mv != PASS && (parity & CELL_DIV4[mv as usize]) != 0 {
                    tmp_moves[n] = mv;
                    tmp_flips[n] = flips[i];
                    n += 1;
                }
            }
            for i in 0..mc {
                let mv = moves[i];
                if mv == PASS || (parity & CELL_DIV4[mv as usize]) == 0 {
                    tmp_moves[n] = mv;
                    tmp_flips[n] = flips[i];
                    n += 1;
                }
            }
            debug_assert!(n == mc);

            moves[..mc].copy_from_slice(&tmp_moves[..mc]);
            flips[..mc].copy_from_slice(&tmp_flips[..mc]);

            // Preserve TT move at the front if present.
            if keep_front != PASS && moves[0] != keep_front {
                if let Some(pos) = moves[..mc].iter().position(|&m| m == keep_front) {
                    moves.swap(0, pos);
                    flips.swap(0, pos);
                }
            }
        } else {
            // Reorder moves only.
            let tmp_moves = &mut self.parity_tmp_moves;
            let mut n = 0usize;

            for i in 0..mc {
                let mv = moves[i];
                if mv != PASS && (parity & CELL_DIV4[mv as usize]) != 0 {
                    tmp_moves[n] = mv;
                    n += 1;
                }
            }
            for i in 0..mc {
                let mv = moves[i];
                if mv == PASS || (parity & CELL_DIV4[mv as usize]) == 0 {
                    tmp_moves[n] = mv;
                    n += 1;
                }
            }
            debug_assert!(n == mc);

            moves[..mc].copy_from_slice(&tmp_moves[..mc]);

            if keep_front != PASS && moves[0] != keep_front {
                if let Some(pos) = moves[..mc].iter().position(|&m| m == keep_front) {
                    moves.swap(0, pos);
                }
            }
        }
    }

    /// Exact endgame solve to completion (depth = empty_count).
    ///
    /// P1-1 implementation:
    /// - PVS / null-window probing on non-PV moves (NWS-style)
    /// - tiny cache-hot local TT storing bounds + best move
    /// - last1/last2 specialized fast paths
    /// - parity-based move ordering (4x4 quadrant parity)
    fn exact_negamax(&mut self, board: &mut Board, alpha: Score, beta: Score, ply: usize) -> (Score, Move) {
        // --- Stage 4 budget check ---
        self.nodes = self.nodes.wrapping_add(1);
        if self.budget_exceeded() {
            self.abort = true;
            return (0, PASS);
        }

        let empty_count = board.empty_count;

        // --- lastN specialized (P1-1 / P1-5) ---
        if empty_count <= 5 {
            let me = board.player;
            let opp = board.opponent;
            let empties = !board.occupied();

            return match empty_count {
                0 => (game_over_scaled(board, board.side), PASS),
                1 => exact_last1(me, opp, empties),
                2 => exact_last2(me, opp, empties),
                3 => exact_last3(me, opp, empties),
                4 => exact_last4(me, opp, empties),
                _ => exact_last5(me, opp, empties),
            };
        }

        let alpha_orig = alpha;
        let beta_orig = beta;
        let mut alpha = alpha;
        let mut beta = beta;

        // Key bitboards (relative to side-to-move) are used for local TT.
        let me_key = board.player;
        let opp_key = board.opponent;

        let mut tt_move: Move = PASS;
        let mut tt_move2: Option<Move> = None;

        // --- P1-1 local TT probe ---
        if let Some(e) = self.end_ltt.probe(empty_count, me_key, opp_key) {
            tt_move = e.best_move;

            if e.lower >= beta {
                return (e.lower, e.best_move);
            }
            if e.upper <= alpha {
                return (e.upper, e.best_move);
            }

            alpha = alpha.max(e.lower);
            beta = beta.min(e.upper);

            if alpha >= beta {
                return (alpha, e.best_move);
            }
        }

        // --- Stage 3 stability cutoff ---
        let mut stability_upper: Option<Score> = None;

        if cfg!(feature = "stability_cutoff") && use_stability_cutoff(empty_count) {
            let (lower, upper) = stability_bounds_for_side_to_move(board);
            let stability_lower = lower.clamp(SCORE_MIN, SCORE_MAX);
            let upper = upper.clamp(SCORE_MIN, SCORE_MAX);
            stability_upper = Some(upper);

            // Cutoffs with stability bounds.
            if stability_lower >= beta {
                return (stability_lower, PASS);
            }
            alpha = alpha.max(stability_lower);

            if upper <= alpha {
                return (upper, PASS);
            }
            beta = beta.min(upper);
        }

        // --- Global TT probe (exact-tagged entries) ---
        let key = board.hash;
        let depth_tag = TT_DEPTH_EXACT_BIAS + empty_count;

        if let Some(e) = self.tt.probe(key) {
            if e.depth >= TT_DEPTH_EXACT_BIAS {
                // Prefer local TT's move if we have one; otherwise use global.
                if tt_move == PASS {
                    tt_move = e.best_move;
                }
                tt_move2 = e.best_move2;

                if e.depth >= depth_tag {
                    match e.flag {
                        Bound::Exact => return (e.value, e.best_move),
                        Bound::Lower => {
                            if e.value >= beta {
                                return (e.value, e.best_move);
                            }
                        }
                        Bound::Upper => {
                            if e.value <= alpha {
                                return (e.value, e.best_move);
                            }
                        }
                    }
                }
            }
        }

        // --- Move generation ---
        let me = me_key;
        let opp = opp_key;
        let mask = legal_moves(me, opp);

        // No moves: PASS or terminal
        if mask == 0 {
            let opp_mask = legal_moves(opp, me);
            if opp_mask == 0 {
                return (game_over_scaled(board, board.side), PASS);
            }

            let undo = {
                let u = &mut self.undo_buf[ply];
                let ok = board.apply_move_no_features(PASS, u);
                debug_assert!(ok);
                *u
            };

            let (child, _mv) = self.exact_negamax(board, -beta, -alpha, ply + 1);

            board.undo_move_no_features(&undo);

            // If we hit the node budget deeper in the tree, unwind without using the partial score.
            if self.abort {
                return (0, PASS);
            }

            return (-child, PASS);
        }

        // --- Stage 2 ordering: dynamically pick a Sensei-style iterator ---
        // In endgame solve, depth_for_mode is the number of empties (pass does
        // not reduce it), matching Sensei's semantics.
        let depth_for_mode = empty_count;
        let mc = self.order_moves_for_node(
            board,
            me,
            opp,
            mask,
            tt_move,
            tt_move2,
            ply,
            depth_for_mode,
            alpha,
            beta,
            stability_upper,
            true, // solve=true (exact endgame)
        );

        // --- P1-1 parity ordering ---
        if mc > 1 && empty_count <= ENDGAME_PARITY_ORDER_MAX_EMPTIES {
            let empties_mask = !board.occupied();
            let parity = parity_bits_from_empties(empties_mask);
            if parity != 0 {
                self.apply_endgame_parity_ordering(ply, mc, parity, tt_move);
            }
        }

        let side_idx = board.side.idx();

        let mut best_score: Score = -INF;
        let mut best_move: Move = PASS;
        let mut best2_score: Score = -INF;
        let mut best2_move: Option<Move> = None;

        for i in 0..mc {
            let mv = self.move_buf[ply][i];
            let cached_flips = if self.flips_valid[ply] { self.flips_buf[ply][i] } else { 0 };

            // P0-7: reuse an Undo slot, and avoid checked make-move overhead for legal moves.
            let undo = {
                let u = &mut self.undo_buf[ply];
                if mv != PASS {
                    if cached_flips != 0 {
                        // Safety: `mv` is a legal move and `cached_flips` were computed for this move.
                        unsafe { board.apply_move_no_features_preflips_unchecked(mv, cached_flips, u) };
                    } else {
                        // Safety: `mv` is a legal move (generated by movegen / ordering).
                        unsafe { board.apply_move_no_features_unchecked(mv, u) };
                    }
                } else {
                    let ok = board.apply_move_no_features(PASS, u);
                    debug_assert!(ok);
                }
                *u
            };

            // TT prefetch (exact endgame): hides cluster latency behind
            // immediate recursion setup.
            self.tt.prefetch(board.hash);

            // P1-1: PVS / NWS-style probing on non-PV moves.
            let score = if i == 0 {
                let (child, _mv2) = self.exact_negamax(board, -beta, -alpha, ply + 1);
                if self.abort {
                    board.undo_move_no_features(&undo);
                    return (0, PASS);
                }
                -child
            } else {
                // Null-window search first.
                let a1 = alpha.saturating_add(1);
                let (child, _mv2) = self.exact_negamax(board, -a1, -alpha, ply + 1);
                if self.abort {
                    board.undo_move_no_features(&undo);
                    return (0, PASS);
                }
                let mut s = -child;

                // If it looks like a PV candidate, re-search with the full window.
                if s > alpha && s < beta {
                    let (child2, _mv3) = self.exact_negamax(board, -beta, -s, ply + 1);
                    if self.abort {
                        board.undo_move_no_features(&undo);
                        return (0, PASS);
                    }
                    s = -child2;
                }
                s
            };

            board.undo_move_no_features(&undo);

            if score > best_score {
                best2_score = best_score;
                best2_move = if best_move == PASS { None } else { Some(best_move) };

                best_score = score;
                best_move = mv;
            } else if mv != best_move && score > best2_score {
                best2_score = score;
                best2_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
            }

            if alpha >= beta {
                self.on_cutoff(ply, side_idx, mv, 1); // depth bonus is irrelevant here
                break;
            }
        }

        let flag = if best_score <= alpha_orig {
            Bound::Upper
        } else if best_score >= beta_orig {
            Bound::Lower
        } else {
            Bound::Exact
        };

        // Store as exact-solver entry (tagged depth).
        self.tt.store(key, depth_tag, flag, best_score, best_move, best2_move);

        // P1-1: also store into the tiny local endgame TT.
        self.end_ltt
            .store(empty_count, me_key, opp_key, flag, best_score, best_move);

        (best_score, best_move)
    }

}

/// Enumerate legal moves bitmask -> out[0..n], no allocation.
#[inline(always)]
fn push_moves_from_mask(mask: u64, out: &mut [Move; MAX_MOVES]) -> usize {
    let mut m = mask;
    let mut n = 0usize;
    while m != 0 {
        if n == MAX_MOVES {
            break;
        }
        let sq = m.trailing_zeros() as u8;
        out[n] = sq;
        n += 1;
        m &= m - 1;
    }
    n
}

/// In-place sort by score descending (mc<=64).
#[inline(always)]
fn sort_moves_desc(moves: &mut [Move; MAX_MOVES], scores: &mut [i32; MAX_MOVES], mc: usize) {
    for i in 1..mc {
        let mv = moves[i];
        let sc = scores[i];

        let mut j = i;
        while j > 0 && scores[j - 1] < sc {
            moves[j] = moves[j - 1];
            scores[j] = scores[j - 1];
            j -= 1;
        }
        moves[j] = mv;
        scores[j] = sc;
    }
}

/// In-place sort by score descending (mc<=64), keeping `flips` aligned with `moves`.
///
/// This is used by ordering modes that precompute flips (P1-1) so the search loop can
/// reuse them on make.
#[inline(always)]
fn sort_moves_desc_with_flips(
    moves: &mut [Move; MAX_MOVES],
    scores: &mut [i32; MAX_MOVES],
    flips: &mut [u64; MAX_MOVES],
    mc: usize,
) {
    for i in 1..mc {
        let mv = moves[i];
        let sc = scores[i];
        let fl = flips[i];

        let mut j = i;
        while j > 0 && scores[j - 1] < sc {
            moves[j] = moves[j - 1];
            scores[j] = scores[j - 1];
            flips[j] = flips[j - 1];
            j -= 1;
        }
        moves[j] = mv;
        scores[j] = sc;
        flips[j] = fl;
    }
}


/// Insert `(mv, score)` into a descending-sorted Top-N vector, truncating to `top_n`.
#[inline(always)]
fn insert_top_n(top: &mut Vec<(Move, Score)>, mv: Move, score: Score, top_n: usize) {
    // Small-N insertion (top_n<=6 in UI), O(N).
    let mut pos = top.len();
    for i in 0..top.len() {
        if score > top[i].1 {
            pos = i;
            break;
        }
    }
    top.insert(pos, (mv, score));
    if top.len() > top_n {
        top.truncate(top_n);
    }
}

/// Convenience free-function entry matching spec:
/// `search(board, alpha, beta, depth) -> (Score, best_move)`.
///
/// Note: this builds a Searcher with zeroed weights. For real engine usage,
/// prefer constructing `Searcher::new(...)` with real weights and a long-lived TT.
pub fn search(board: &mut Board, alpha: Score, beta: Score, depth: u8) -> (Score, Move) {
    let weights = Weights::zeroed();
    let (feats, occ) = build_sonetto_feature_defs_and_occ();
    let swap = SwapTables::build_swap_tables();

    let mut s = Searcher::new(8, weights, feats, swap, occ); // 8MB TT default
    s.search(board, alpha, beta, depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_legal_move(board: &Board, mv: Move) -> bool {
        if mv == PASS {
            return true;
        }
        let me = board.player;
        let opp = board.opponent;
        let mask = legal_moves(me, opp);
        (mask & (1u64 << (mv as u64))) != 0
    }

    #[test]
    fn search_start_position_depth_4_returns_legal_opening_move() {
        let mut b = Board::new_start(0);
        let (s, mv) = search(&mut b, -INF, INF, 4);
        let _ = s; // score not asserted here

        assert!(mv != PASS, "start position should have moves");
        assert!(is_legal_move(&b, mv), "best_move must be legal");

        // Standard opening moves in row-major bitpos: 19,26,37,44
        assert!([19u8, 26u8, 37u8, 44u8].contains(&mv));
    }

    #[test]
    fn search_after_one_move_returns_legal_reply() {
        let mut b = Board::new_start(0);

        // Apply a common opening move if legal.
        let mut u = Undo::default();
        if !b.apply_move(19u8, &mut u) {
            return;
        }

        let (s, mv) = search(&mut b, -INF, INF, 4);
        let _ = s;
        assert!(mv == PASS || is_legal_move(&b, mv));
    }

    #[test]
    fn search_does_not_corrupt_board_state() {
        let mut b = Board::new_start(0);
        let before = (b.player, b.opponent, b.side, b.empty_count, b.hash);

        let _ = search(&mut b, -INF, INF, 4);

        let after = (b.player, b.opponent, b.side, b.empty_count, b.hash);
        assert_eq!(before, after, "search must leave board unchanged");
    }
}
