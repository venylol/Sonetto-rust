//! Sensei-style move ordering iterators.
//!
//! These are inspired by OthelloSensei's `MoveIterator*` family used by
//! `EvaluatorAlphaBeta`.
//!
//! The goal is not a byte-for-byte port, but to preserve the *shape* of the
//! move ordering so the alpha-beta backend behaves similarly and can be
//! optimized later.
//!
//! Notes:
//! - We use Sonetto's `flips_for_move` implementation; its flips mask **does not
//!   include** the move bit.
//! - Some iterators start from an over-approx candidate set and filter by `flips_for_move != 0`;
//!   others use `legal_moves` directly.
//! - Disproof-number ordering uses the ported Sensei EndgameTime regression model
//!   (`sensei_extras::endgame_time`).

use crate::board::Color;
use crate::coord::Move;
use crate::eval::{
    abs_to_egev2_idx, phase, round_clamp_raw_to_disc, EGEV2_FEATURE_TO_COORD, FEATURE_TO_PATTERN,
    N_PATTERN_FEATURES, PARAMS_PER_PHASE, PATTERN_OFFSETS, PATTERN_PARAMS_PER_PHASE, Weights,
};
use crate::features::occ::{OccMap, Occurrence};
use crate::flips::flips_for_move;
use crate::movegen::{legal_moves, NOT_A, NOT_H};
use crate::score::Score;
use crate::sensei_extras::endgame_time;

use super::depth_one::{apply_move_feat_ids, rollback_move_feat_ids, score_disc_from_abs_ids};

/// Move ordering mode (Sensei-inspired).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveIteratorKind {
    /// Equivalent of Sensei's `MoveIteratorVeryQuick`.
    VeryQuick,
    /// Equivalent of Sensei's `MoveIteratorQuick<true>`.
    Quick1,
    /// Equivalent of Sensei's `MoveIteratorQuick<false>`.
    Quick2,
    /// Equivalent of Sensei's `MoveIteratorMinimizeOpponentMoves`.
    MinimizeOpponentMoves,
    /// Equivalent of Sensei's `MoveIteratorDisproofNumber`.
    DisproofNumber,
}

/// Additional context for Sensei-style disproof-number ordering.
///
/// When provided, we can compute a faithful depth-one style approximate evaluation
/// of the child position using **incremental absolute pattern IDs** (EGEV2).
/// This replaces the older "disc-diff only" approximation.
pub struct DisproofCtx<'a> {
    pub side: Color,
    pub empty_count: u8,
    pub player_discs: u8,
    pub opponent_discs: u8,
    pub feat_id_abs: &'a mut [u16],
    pub occ: &'a OccMap,
    pub weights: &'a Weights,
}

/// Corners (A1, H1, A8, H8) in Sonetto internal bitpos.
pub const CORNER_MASK: u64 = 0x8100_0000_0000_0081;

/// "X" squares (diagonal-adjacent to corners) in Sonetto internal bitpos.
///
/// - B2 (9), G2 (14), B7 (49), G7 (54)
pub const X_MASK: u64 = 0x0042_0000_0000_4200;

/// Sensei's `kCentralPattern`, translated to Sonetto's bitpos.
const CENTRAL_PATTERN: u64 = 0x003c_7e7e_7e7e_3c00;

/// Sensei's `kEdgePattern`, translated to Sonetto's bitpos.
const EDGE_PATTERN: u64 = 0x3c00_8181_8181_003c;

const FIRST_ROW: u64 = 0xff00_0000_0000_0000;
const LAST_ROW: u64 = 0x0000_0000_0000_00ff;
const FIRST_COL: u64 = 0x0101_0101_0101_0101;
const LAST_COL: u64 = 0x8080_8080_8080_8080;

/// Sensei square ordering values (`kSquareValue`).
///
/// Symmetric across both axes, so we can reuse it directly under Sonetto's
/// internal mapping.
const SQUARE_VALUE: [i32; 64] = [
    18, 4, 16, 12, 12, 16, 4, 18, // 1
    4, 2, 6, 8, 8, 6, 2, 4, // 2
    16, 6, 14, 10, 10, 14, 6, 16, // 3
    12, 8, 10, 0, 0, 10, 8, 12, // 4
    12, 8, 10, 0, 0, 10, 8, 12, // 5
    16, 6, 14, 10, 10, 14, 6, 16, // 6
    4, 2, 6, 8, 8, 6, 2, 4, // 7
    18, 4, 16, 12, 12, 16, 4, 18, // 8
];

#[inline(always)]
fn neighbors(x: u64) -> u64 {
    // Coordinate system: A1=0, H1=7, A8=56.
    let n = x << 8;
    let s = x >> 8;
    let e = (x & NOT_H) << 1;
    let w = (x & NOT_A) >> 1;
    let ne = (x & NOT_H) << 9;
    let nw = (x & NOT_A) << 7;
    let se = (x & NOT_H) >> 7;
    let sw = (x & NOT_A) >> 9;
    n | s | e | w | ne | nw | se | sw
}

#[inline(always)]
fn unique_set(b: u64) -> u64 {
    if b != 0 && (b & (b - 1)) == 0 { b } else { 0 }
}

#[inline(always)]
fn first_last_set(b: u64) -> u64 {
    if b == 0 {
        0
    } else {
        let lo = b & b.wrapping_neg();
        // `leading_zeros` returns `u32`, which is a valid shift amount.
        let hi = 1u64 << (63 - b.leading_zeros());
        lo | hi
    }
}

#[inline(always)]
fn unique_in_edges(empties: u64) -> u64 {
    unique_set(empties & FIRST_ROW)
        | unique_set(empties & LAST_ROW)
        | unique_set(empties & FIRST_COL)
        | unique_set(empties & LAST_COL)
}

#[inline(always)]
fn first_last_in_edges(empties: u64) -> u64 {
    (first_last_set(empties & FIRST_ROW)
        | first_last_set(empties & LAST_ROW)
        | first_last_set(empties & FIRST_COL)
        | first_last_set(empties & LAST_COL))
        & !CORNER_MASK
}

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    mv: Move,
    mv_bit: u64,
    flips: u64,
    value: i32,
}

/// Fill `out_moves/out_flips` with ordered legal moves, returning how many.
///
/// `last_flip_incl_move` should be `move_bit | flips` from the previous ply.
/// (Matches Sensei's convention; Sonetto's `Undo.flips` does not include the
/// move bit.)
#[inline]
pub fn gen_ordered_moves(
    kind: MoveIteratorKind,
    player: u64,
    opponent: u64,
    last_flip_incl_move: u64,
    beta: Score,
    mut disproof_ctx: Option<DisproofCtx<'_>>,
    out_moves: &mut [Move; 64],
    out_flips: &mut [u64; 64],
) -> usize {
    match kind {
        MoveIteratorKind::VeryQuick => gen_very_quick(player, opponent, out_moves, out_flips),
        MoveIteratorKind::Quick1 => gen_quick::<true>(
            player,
            opponent,
            last_flip_incl_move,
            out_moves,
            out_flips,
        ),
        MoveIteratorKind::Quick2 => gen_quick::<false>(
            player,
            opponent,
            last_flip_incl_move,
            out_moves,
            out_flips,
        ),
        MoveIteratorKind::MinimizeOpponentMoves => {
            gen_minimize_opponent_moves(player, opponent, out_moves, out_flips)
        }
        MoveIteratorKind::DisproofNumber => {
            gen_disproof_number(player, opponent, beta, disproof_ctx.as_mut(), out_moves, out_flips)
        }
    }
}

#[inline]
fn gen_very_quick(player: u64, opponent: u64, out_moves: &mut [Move; 64], out_flips: &mut [u64; 64]) -> usize {
    let empties = !(player | opponent);
    let mut candidates = neighbors(opponent) & empties;

    let mut n = 0usize;
    while candidates != 0 {
        let mv_bit = candidates & candidates.wrapping_neg();
        candidates &= candidates - 1;

        let flips = flips_for_move(player, opponent, mv_bit);
        if flips == 0 {
            continue;
        }

        out_moves[n] = mv_bit.trailing_zeros() as Move;
        out_flips[n] = flips;
        n += 1;
    }

    n
}

#[inline]
fn gen_quick<const VERY_QUICK: bool>(
    player: u64,
    opponent: u64,
    last_flip_incl_move: u64,
    out_moves: &mut [Move; 64],
    out_flips: &mut [u64; 64],
) -> usize {
    let empties = !(player | opponent);
    let neighbors_player = neighbors(player);

    // Sensei: candidates = Neighbors(opponent) & empties.
    let mut candidates = neighbors(opponent) & empties;

    // Sensei's `masks_` array.
    let mut masks = [0u64; 9];
    let mask_len: usize;

    if VERY_QUICK {
        // MoveIteratorQuick<true>
        masks[1] = CORNER_MASK;
        masks[3] = CENTRAL_PATTERN;
        masks[4] = EDGE_PATTERN;
        masks[5] = !0u64;
        mask_len = 6;

        masks[0] = !neighbors(empties) & neighbors_player;
        masks[2] = if (last_flip_incl_move & X_MASK) != 0 {
            neighbors(last_flip_incl_move)
        } else {
            0
        };
    } else {
        // MoveIteratorQuick<false>
        masks[3] = CORNER_MASK;
        masks[6] = CENTRAL_PATTERN;
        masks[7] = EDGE_PATTERN;
        masks[8] = !0u64;
        mask_len = 9;

        masks[0] = !neighbors(empties) & neighbors_player;
        masks[1] = unique_in_edges(empties) & neighbors_player;
        masks[2] = neighbors(last_flip_incl_move) & CORNER_MASK;
        masks[4] = first_last_in_edges(empties);
        masks[5] = if (last_flip_incl_move & X_MASK) != 0 {
            neighbors(last_flip_incl_move)
        } else {
            0
        };
    }

    let mut n = 0usize;
    let mut current_mask: usize = 0;

    while candidates != 0 {
        while current_mask + 1 < mask_len && (candidates & masks[current_mask]) == 0 {
            current_mask += 1;
        }

        let picked = candidates & masks[current_mask];
        // `masks[last]` is `!0`, so picked must be non-zero when candidates != 0.
        debug_assert!(picked != 0);

        let mv_bit = picked & picked.wrapping_neg();
        candidates &= !mv_bit;

        let flips = flips_for_move(player, opponent, mv_bit);
        if flips == 0 {
            continue;
        }

        out_moves[n] = mv_bit.trailing_zeros() as Move;
        out_flips[n] = flips;
        n += 1;
    }

    n
}

#[inline]
fn gen_minimize_opponent_moves(
    player: u64,
    opponent: u64,
    out_moves: &mut [Move; 64],
    out_flips: &mut [u64; 64],
) -> usize {
    let empties = !(player | opponent);
    let mut candidates = neighbors(opponent) & empties;

    let mut entries = [Entry::default(); 64];
    let mut n = 0usize;

    while candidates != 0 {
        let mv_bit = candidates & candidates.wrapping_neg();
        candidates &= candidates - 1;

        let flips = flips_for_move(player, opponent, mv_bit);
        if flips == 0 {
            continue;
        }

        let mv = mv_bit.trailing_zeros() as Move;

        // After the move, the *opponent* becomes side-to-move.
        let new_player = opponent & !flips;
        let new_opp = player | mv_bit | flips;

        let opp_moves = legal_moves(new_player, new_opp);
        let mobility = opp_moves.count_ones() as i32;
        let corner_mobility = (opp_moves & CORNER_MASK).count_ones() as i32;

        let value = -((mobility + corner_mobility) * 1000) + SQUARE_VALUE[mv as usize];

        entries[n] = Entry {
            mv,
            mv_bit,
            flips,
            value,
        };
        n += 1;
    }

    // Selection-sort style extraction: repeatedly pick max.
    let mut out_n = 0usize;
    while n != 0 {
        let mut best_i = 0usize;
        let mut best_v = entries[0].value;
        for i in 1..n {
            let v = entries[i].value;
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }

        let best = entries[best_i];
        entries[best_i] = entries[n - 1];
        n -= 1;

        out_moves[out_n] = best.mv;
        out_flips[out_n] = best.flips;
        out_n += 1;
    }

    out_n
}

#[inline]

#[inline(always)]
fn raw_pattern_sum_from_abs_ids(eval_side: Color, phase_base: usize, params: &[i16], feat_id_abs: &[u16]) -> i32 {
    debug_assert!(feat_id_abs.len() >= N_PATTERN_FEATURES);
    let mut raw: i32 = 0;

    // NOTE: This is intentionally hot; keep bounds checks out of the loop.
    for fi in 0..N_PATTERN_FEATURES {
        unsafe {
            let abs_id = *feat_id_abs.get_unchecked(fi);
            let len = EGEV2_FEATURE_TO_COORD.get_unchecked(fi).n_cells as usize;
            let idx = abs_to_egev2_idx(eval_side, len, abs_id);

            let pat = *FEATURE_TO_PATTERN.get_unchecked(fi) as usize;
            let base = phase_base + *PATTERN_OFFSETS.get_unchecked(pat);

            raw += *params.get_unchecked(base + idx) as i32;
        }
    }

    raw
}

#[inline(always)]
unsafe fn update_feat_for_sq_with_raw_pattern(
    feat_id_abs: &mut [u16],
    raw_pattern: &mut i32,
    occ: &OccMap,
    flat: &[Occurrence],
    sq: Move,
    delta_digit: i32,
    eval_side: Color,
    phase_base: usize,
    params: &[i16],
) {
    let (start, end) = occ.range_for_sq(sq);

    let occs = flat.get_unchecked(start..end);
    for occ in occs {
        let fi = occ.feature_idx as usize;
        let pow3 = occ.pow3 as i32;

        let old_abs = *feat_id_abs.get_unchecked(fi) as i32;
        let new_abs = old_abs + delta_digit * pow3;

        *feat_id_abs.get_unchecked_mut(fi) = new_abs as u16;

        let len = EGEV2_FEATURE_TO_COORD.get_unchecked(fi).n_cells as usize;
        let pat = *FEATURE_TO_PATTERN.get_unchecked(fi) as usize;
        let base = phase_base + *PATTERN_OFFSETS.get_unchecked(pat);

        let old_idx = abs_to_egev2_idx(eval_side, len, old_abs as u16);
        let new_idx = abs_to_egev2_idx(eval_side, len, new_abs as u16);

        let w_old = *params.get_unchecked(base + old_idx) as i32;
        let w_new = *params.get_unchecked(base + new_idx) as i32;

        *raw_pattern += w_new - w_old;
    }
}

#[inline(always)]
unsafe fn apply_move_feat_ids_with_raw_pattern(
    feat_id_abs: &mut [u16],
    raw_pattern: &mut i32,
    occ: &OccMap,
    flat: &[Occurrence],
    mv: Move,
    flips: u64,
    mover: Color,
    eval_side: Color,
    phase_base: usize,
    params: &[i16],
) {
    let mover_digit_abs: i32 = if mover == Color::Black { 1 } else { 2 };
    update_feat_for_sq_with_raw_pattern(
        feat_id_abs,
        raw_pattern,
        occ,
        flat,
        mv,
        mover_digit_abs,
        eval_side,
        phase_base,
        params,
    );

    let flip_delta: i32 = if mover == Color::Black { -1 } else { 1 };
    let mut bb = flips;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        update_feat_for_sq_with_raw_pattern(
            feat_id_abs,
            raw_pattern,
            occ,
            flat,
            sq,
            flip_delta,
            eval_side,
            phase_base,
            params,
        );
    }
}

#[inline(always)]
unsafe fn rollback_move_feat_ids_with_raw_pattern(
    feat_id_abs: &mut [u16],
    raw_pattern: &mut i32,
    occ: &OccMap,
    flat: &[Occurrence],
    mv: Move,
    flips: u64,
    mover: Color,
    eval_side: Color,
    phase_base: usize,
    params: &[i16],
) {
    let mover_digit_abs: i32 = if mover == Color::Black { 1 } else { 2 };
    update_feat_for_sq_with_raw_pattern(
        feat_id_abs,
        raw_pattern,
        occ,
        flat,
        mv,
        -mover_digit_abs,
        eval_side,
        phase_base,
        params,
    );

    let flip_delta: i32 = if mover == Color::Black { -1 } else { 1 };
    let mut bb = flips;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        update_feat_for_sq_with_raw_pattern(
            feat_id_abs,
            raw_pattern,
            occ,
            flat,
            sq,
            -flip_delta,
            eval_side,
            phase_base,
            params,
        );
    }
}


#[inline(always)]
fn prefer_better_move_order(a_mv_bit: u64, b_mv_bit: u64) -> bool {
    // Deterministic tie-breaker: prefer lower square index.
    a_mv_bit < b_mv_bit
}

fn gen_disproof_number(
    player: u64,
    opponent: u64,
    beta: Score,
    mut ctx: Option<&mut DisproofCtx>,
    out_moves: &mut [Move; 64],
    out_flips: &mut [u64; 64],
) -> usize {
    let upper_eval_large = (beta >> 2).clamp(-512, 512);
    let lower_eval_large = -upper_eval_large;

    let mut mask = legal_moves(player, opponent);

    // Gather candidates (unordered for now).
    let mut entries = [Entry {
        mv: 0,
        mv_bit: 0,
        flips: 0,
        value: 0,
    }; 64];
    let mut n = 0usize;

    if let Some(c) = ctx.as_deref_mut() {
        let mover = c.side;
        let child_side = mover.other();
        let child_empty = c.empty_count.saturating_sub(1);

        // Split borrows: we mutate `feat_id_abs`, but `occ`/`weights` are read-only.
        let feat_id_abs: &mut [u16] = &mut *c.feat_id_abs;
        let occ: &OccMap = c.occ;
        let weights: &Weights = c.weights;
        debug_assert!(feat_id_abs.len() == N_PATTERN_FEATURES);

        // -------- Approx-eval cache (incremental) --------
        // We evaluate *child* positions, so use the child's side and empty-count/phase.
        let params = weights.as_slice();
        let phase_base = phase(child_empty) * PARAMS_PER_PHASE;
        let phase_end = phase_base + PARAMS_PER_PHASE;
        let eval_ok = phase_end <= params.len();

        let eval_num_base = phase_base + PATTERN_PARAMS_PER_PHASE;

        let flat = occ.flat();
        let mut raw_pattern = if eval_ok {
            raw_pattern_sum_from_abs_ids(child_side, phase_base, params, feat_id_abs)
        } else {
            0
        };

        while mask != 0 {
            let mv = mask.trailing_zeros() as u8;
            mask &= mask - 1;

            let mv_bit = 1u64 << mv;
            let flips = flips_for_move(player, opponent, mv_bit);

            // Child bitboards (player/opponent already swapped).
            let new_player = opponent & !flips;
            let new_opp = player | mv_bit | flips;

            // --- Approx eval (EvalLarge: disc * 8) ---
            let approx_eval_large: i32 = if eval_ok {
                let flips_cnt = flips.count_ones() as u8;
                let child_player_discs = c.opponent_discs.saturating_sub(flips_cnt);

                unsafe {
                    apply_move_feat_ids_with_raw_pattern(
                        feat_id_abs,
                        &mut raw_pattern,
                        occ,
                        flat,
                        mv,
                        flips,
                        mover,
                        child_side,
                        phase_base,
                        params,
                    );
                }

                let eval_num = unsafe {
                    *params.get_unchecked(eval_num_base + child_player_discs as usize) as i32
                };
                let disc = round_clamp_raw_to_disc(raw_pattern + eval_num);
                let out = disc * 8;

                unsafe {
                    rollback_move_feat_ids_with_raw_pattern(
                        feat_id_abs,
                        &mut raw_pattern,
                        occ,
                        flat,
                        mv,
                        flips,
                        mover,
                        child_side,
                        phase_base,
                        params,
                    );
                }

                out
            } else {
                // Fallback: full feature-sum eval (still correct, just slower).
                let flips_cnt = flips.count_ones() as u8;
                let child_player_discs = c.opponent_discs.saturating_sub(flips_cnt);

                apply_move_feat_ids(feat_id_abs, occ, mv, flips, mover);
                let disc =
                    score_disc_from_abs_ids(child_side, child_empty, child_player_discs, feat_id_abs, weights);
                rollback_move_feat_ids(feat_id_abs, occ, mv, flips, mover);

                disc * 8
            };

            let dn = endgame_time::disproof_number_over_prob(new_player, new_opp, lower_eval_large, approx_eval_large);
            let value = -dn + SQUARE_VALUE[mv as usize];

            entries[n] = Entry {
                mv,
                mv_bit,
                flips,
                value,
            };
            n += 1;
        }
    } else {
        // No eval context => fall back to a pure endgame-time heuristic (disc-count only).
        while mask != 0 {
            let mv = mask.trailing_zeros() as u8;
            mask &= mask - 1;

            let mv_bit = 1u64 << mv;
            let flips = flips_for_move(player, opponent, mv_bit);

            let new_player = opponent & !flips;
            let new_opp = player | mv_bit | flips;

            let approx_eval_large: i32 = (new_player.count_ones() as i32 - new_opp.count_ones() as i32) * 8;

            let dn = endgame_time::disproof_number_over_prob(new_player, new_opp, lower_eval_large, approx_eval_large);
            let value = -dn + SQUARE_VALUE[mv as usize];

            entries[n] = Entry {
                mv,
                mv_bit,
                flips,
                value,
            };
            n += 1;
        }
    }

    // Select the best move first, then the next best, etc.
    // (We keep this structure to match Sensei's original iterator behavior.)
    let mut used = 0usize;
    while used < n {
        let mut best_i = used;
        let mut best_v = entries[used].value;

        for i in (used + 1)..n {
            let v = entries[i].value;
            if v > best_v || (v == best_v && prefer_better_move_order(entries[i].mv_bit, entries[best_i].mv_bit)) {
                best_i = i;
                best_v = v;
            }
        }

        entries.swap(used, best_i);

        out_moves[used] = entries[used].mv;
        out_flips[used] = entries[used].flips;
        used += 1;
    }

    n
}
