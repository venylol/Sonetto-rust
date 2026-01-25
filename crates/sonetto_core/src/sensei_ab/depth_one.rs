//! Sensei-style "depth-one" evaluator helpers for the EGEV2 evaluation pipeline.
//!
//! Native Sensei maintains an incremental pattern evaluator that can be updated/undone
//! with just `(move_square, flip_mask)` (without mutating the full board object).
//!
//! In Sonetto we already track **absolute ternary pattern IDs** for the 64 EGEV2
//! symmetry-expanded features. That makes the update part cheap; we only need:
//! - a fast "apply deltas" helper for `feat_id_abs`
//! - a fast evaluation path that consumes `(side, empty_count, num_player_discs, feat_ids)`
//!   without depending on full bitboards.

use crate::board::Color;
use crate::coord::{Move, PASS};
use crate::eval::{
    abs_to_egev2_idx, phase, round_clamp_raw_to_disc, FeatureToCoord, Weights, EGEV2_FEATURE_TO_COORD,
    FEATURE_TO_PATTERN, N_PATTERN_FEATURES, EVAL_NUM_LEN, EXPECTED_PARAMS_LEN, PARAMS_PER_PHASE,
    PATTERN_OFFSETS, PATTERN_PARAMS_PER_PHASE, PATTERN_POW3,
};
use crate::features::occ::{OccMap, Occurrence};

/// Apply a digit delta (in { -2..=2 }) to all feature IDs that include `sq`.
///
/// This is the same hot-path logic used by `features::update`, but exposed here so
/// the Sensei AB backend can update a standalone `[u16; 64]` feature-id cache.
#[inline(always)]
fn update_feat_for_sq_fast(feat: &mut [u16], occ: &OccMap, flat: &[Occurrence], sq: u8, delta: i32) {
    debug_assert_eq!(feat.len(), N_PATTERN_FEATURES);
    debug_assert!(sq < 64);
    if delta == 0 {
        return;
    }

    let (s, e) = occ.range_for_sq(sq);
    let len = e - s;

    // Safety: OccMap for EGEV2 must only reference feature indices < 64.
    macro_rules! step {
        ($k:expr) => {{
            let o = unsafe { *flat.get_unchecked(s + $k) };
            let idx = o.feature_idx as usize;
            debug_assert!(idx < N_PATTERN_FEATURES);

            let cur = unsafe { *feat.get_unchecked(idx) as i32 };
            let pow3 = o.pow3 as i32;
            let add = match delta {
                1 => pow3,
                -1 => -pow3,
                2 => pow3 << 1,
                -2 => -(pow3 << 1),
                _ => delta * pow3,
            };
            let nxt = cur + add;

            debug_assert!(nxt >= 0);
            debug_assert!(nxt <= (u16::MAX as i32));
            unsafe { *feat.get_unchecked_mut(idx) = nxt as u16 };
        }};
    }

    match len {
        0 => {}
        1 => { step!(0); }
        2 => { step!(0); step!(1); }
        3 => { step!(0); step!(1); step!(2); }
        4 => { step!(0); step!(1); step!(2); step!(3); }
        5 => { step!(0); step!(1); step!(2); step!(3); step!(4); }
        6 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); }
        7 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); }
        8 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); }
        9 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); }
        10 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); }
        11 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); }
        12 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); }
        13 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); }
        14 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); }
        15 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); step!(14); }
        16 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); step!(14); step!(15); }
        _ => {
            let mut i = s;
            while i < e {
                let o = unsafe { *flat.get_unchecked(i) };
                let idx = o.feature_idx as usize;
                debug_assert!(idx < N_PATTERN_FEATURES);

                let cur = unsafe { *feat.get_unchecked(idx) as i32 };
                let pow3 = o.pow3 as i32;
                let add = match delta {
                    1 => pow3,
                    -1 => -pow3,
                    2 => pow3 << 1,
                    -2 => -(pow3 << 1),
                    _ => delta * pow3,
                };
                let nxt = cur + add;

                debug_assert!(nxt >= 0);
                debug_assert!(nxt <= (u16::MAX as i32));

                unsafe { *feat.get_unchecked_mut(idx) = nxt as u16 };
                i += 1;
            }
        }
    }
}

/// Apply the absolute-pattern-id updates for a move (EGEV2).
///
/// - `mv` is in internal bitpos (0..63)
/// - `flips` does **not** include the move bit
/// - `mover` is the absolute color of the mover
#[inline(always)]
pub(crate) fn apply_move_feat_ids(
    feat: &mut [u16],
    occ: &OccMap,
    mv: Move,
    flips: u64,
    mover: Color,
) {
    debug_assert_eq!(feat.len(), N_PATTERN_FEATURES);
    if mv == PASS {
        return;
    }
    let flat = occ.flat();

    // placed square: empty -> mover digit (Black=1, White=2)
    let placed_delta: i32 = mover.digit_abs() as i32;
    update_feat_for_sq_fast(feat, occ, flat, mv, placed_delta);

    // flips: opponent -> mover
    let flip_delta: i32 = match mover {
        Color::Black => -1, // White(2) -> Black(1)
        Color::White => 1,  // Black(1) -> White(2)
    };

    let mut f = flips;
    while f != 0 {
        let sq = f.trailing_zeros() as u8;
        f &= f - 1;
        update_feat_for_sq_fast(feat, occ, flat, sq, flip_delta);
    }
}

/// Roll back the absolute-pattern-id updates for a move.
#[inline(always)]
pub(crate) fn rollback_move_feat_ids(
    feat: &mut [u16],
    occ: &OccMap,
    mv: Move,
    flips: u64,
    mover: Color,
) {
    debug_assert_eq!(feat.len(), N_PATTERN_FEATURES);
    if mv == PASS {
        return;
    }
    let flat = occ.flat();
    let placed_delta: i32 = -(mover.digit_abs() as i32);
    update_feat_for_sq_fast(feat, occ, flat, mv, placed_delta);

    let flip_delta: i32 = match mover {
        Color::Black => -1,
        Color::White => 1,
    };
    let mut f = flips;
    while f != 0 {
        let sq = f.trailing_zeros() as u8;
        f &= f - 1;
        update_feat_for_sq_fast(feat, occ, flat, sq, -flip_delta);
    }
}

/// Fast EGEV2 disc-score evaluation from absolute feature IDs.
///
/// This mirrors the `score_disc()` fast path, but avoids depending on full
/// bitboards (only `num_player_discs` is needed for `eval_num`).
#[inline(always)]
pub(crate) fn score_disc_from_abs_ids(
    side: Color,
    empty_count: u8,
    num_player_discs: u8,
    feat: &[u16],
    weights: &Weights,
) -> i32 {
    debug_assert_eq!(feat.len(), N_PATTERN_FEATURES);
    let ph = phase(empty_count);

    // Same layout/fast path as eval_egev2::score_disc.
    let params = weights.as_slice();
    let valid = params.len() == EXPECTED_PARAMS_LEN;

    let mut raw: i32 = 0;
    if valid {
        let phase_base = ph * PARAMS_PER_PHASE;
        let mut fi = 0usize;
        while fi < N_PATTERN_FEATURES {
            let pattern = FEATURE_TO_PATTERN[fi];
            let FeatureToCoord { n_cells, .. } = EGEV2_FEATURE_TO_COORD[fi];
            let len = n_cells as usize;
            let abs_id = feat[fi];
            let idx = abs_to_egev2_idx(side, len, abs_id);
            debug_assert!(idx < PATTERN_POW3[pattern]);
            let off = phase_base + PATTERN_OFFSETS[pattern] + idx;
            raw += unsafe { *params.get_unchecked(off) } as i32;
            fi += 1;
        }

        let nd = num_player_discs as usize;
        debug_assert!(nd < EVAL_NUM_LEN);
        let off = phase_base + PATTERN_PARAMS_PER_PHASE + nd;
        raw += unsafe { *params.get_unchecked(off) } as i32;
        return round_clamp_raw_to_disc(raw);
    }

    // Fallback: missing/partial weights.
    let mut fi = 0usize;
    while fi < N_PATTERN_FEATURES {
        let pattern = FEATURE_TO_PATTERN[fi];
        let FeatureToCoord { n_cells, .. } = EGEV2_FEATURE_TO_COORD[fi];
        let len = n_cells as usize;
        let abs_id = feat[fi];
        let idx = abs_to_egev2_idx(side, len, abs_id);
        raw += weights.get_pattern_weight(ph, pattern, idx) as i32;
        fi += 1;
    }
    raw += weights.get_eval_num_weight(ph, num_player_discs as usize) as i32;
    round_clamp_raw_to_disc(raw)
}
