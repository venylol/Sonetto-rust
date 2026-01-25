// crates/sonetto-core/src/features/update.rs

use crate::board::{Board, Color, Undo};
use crate::coord::PASS;
use crate::eval::N_PATTERN_FEATURES;

use super::occ::OccMap;

/// Apply a digit delta (in { -2..=2 }) on all feature IDs that include `sq`.
///
/// Feature IDs are base-3 numbers in absolute digits:
/// - empty = 0
/// - black = 1
/// - white = 2
///
/// The increment for a particular feature occurrence is `delta * pow3`.
#[inline(always)]
pub fn update_feat_for_sq(board: &mut Board, occ: &OccMap, sq: u8, delta: i32) {
    debug_assert!(sq < 64);
    if delta == 0 {
        return;
    }
    for o in occ.occ_for_sq(sq) {
        let idx = o.feature_idx as usize;
        // Safety: OccMap must be consistent with `feat_id_abs` length, but we
        // guard anyway to avoid panics on malformed feature metadata.
        if idx >= board.feat_id_abs.len() {
            continue;
        }

        let cur = board.feat_id_abs[idx] as i32;
        let nxt = cur + delta * (o.pow3 as i32);

        // Safety: in a correct engine `nxt` is always non-negative. If
        // something goes wrong upstream, clamp to 0 to avoid producing a huge
        // bogus u32 id that could later cause OOB access in swap/weights.
        board.feat_id_abs[idx] = if nxt <= 0 { 0u16 } else if nxt >= (u16::MAX as i32) { u16::MAX } else { nxt as u16 };
    }
}

// ---------------------------------------------------------------------------
// P0-3 fast path: pattern-id incremental updates (hot in negamax)
// ---------------------------------------------------------------------------

/// Update a single square in the pattern-id cache using unchecked indexing.
///
/// Preconditions:
/// - `feat.len() == N_PATTERN_FEATURES` (64)
/// - `sq < 64`
/// - `occ` is consistent (all `feature_idx < N_PATTERN_FEATURES`)
#[inline(always)]
fn update_feat_for_sq_fast(feat: &mut [u16], occ: &OccMap, flat: &[super::occ::Occurrence], sq: u8, delta: i32) {
    debug_assert!(sq < 64);
    if delta == 0 {
        return;
    }

    let (s, e) = occ.range_for_sq(sq);
    let len = e - s;

    // ---------------------------------------------------------------------
    // P1-4: OccMap update unrolling
    // ---------------------------------------------------------------------
    //
    // Hot-path feature updates are called once for the placed move and once for
    // every flipped disc. For EGEV2, each square participates in only a small
    // number of features, so we can shave a few cycles by unrolling the common
    // small occurrence counts and avoiding the loop branch + index bump.
    //
    // For large/custom feature sets (tests/tooling), we fall back to the
    // original loop.

    // Safety: caller guarantees OccMap indices are in-bounds for `feat`.
    macro_rules! step {
        ($k:expr) => {
            // Copy (Occurrence is Copy) to avoid borrow issues when mutating `feat`.
            let o = unsafe { *flat.get_unchecked(s + $k) };
            let idx = o.feature_idx as usize;

            // Compute next id with tiny-delta specialization to avoid a multiply.
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

            unsafe {
                *feat.get_unchecked_mut(idx) = nxt as u16;
            }
        };
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
                // Copy (Occurrence is Copy) to avoid borrow issues when mutating `feat`.
                let o = unsafe { *flat.get_unchecked(i) };
                let idx = o.feature_idx as usize;

                // Compute next id with tiny-delta specialization to avoid a multiply.
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

                unsafe {
                    *feat.get_unchecked_mut(idx) = nxt as u16;
                }

                i += 1;
            }
        }
    }
}

/// Incrementally update feature IDs for a move that has already been applied to the bitboards.
///
/// Required semantics:
/// - placed square: empty -> mover digit (Black=1, White=2)
/// - flips: opponent -> mover
/// - flip_delta: Black = -1 (2->1), White = +1 (1->2)
#[inline(always)]
pub fn update_features_for_move(board: &mut Board, undo: &Undo, occ: &OccMap) {
    if undo.mv == PASS {
        return;
    }

    // Hot path during search: `feat_id_abs` holds absolute ternary pattern IDs.
    // This matches Egaroucid/Sensei style incremental pattern bookkeeping.
    if board.feat_is_pattern_ids && board.feat_id_abs.len() == N_PATTERN_FEATURES {
        let flat = occ.flat();
        let mover = undo.old_side;
        let placed_delta = mover.digit_abs() as i32;
        update_feat_for_sq_fast(&mut board.feat_id_abs, occ, flat, undo.mv, placed_delta);

        let flip_delta: i32 = match mover {
            Color::Black => -1,
            Color::White => 1,
        };

        let mut f = undo.flips;
        while f != 0 {
            let sq = f.trailing_zeros() as u8;
            f &= f - 1;
            update_feat_for_sq_fast(&mut board.feat_id_abs, occ, flat, sq, flip_delta);
        }
        return;
    }

    let mover = undo.old_side;
    let placed_delta = mover.digit_abs() as i32;
    update_feat_for_sq(board, occ, undo.mv, placed_delta);

    let flip_delta: i32 = match mover {
        Color::Black => -1,
        Color::White => 1,
    };

    let mut f = undo.flips;
    while f != 0 {
        let sq = f.trailing_zeros() as u8;
        f &= f - 1;
        update_feat_for_sq(board, occ, sq, flip_delta);
    }
}

/// Roll back feature IDs for a move that is going to be undone.
///
/// This is the exact inverse of [`update_features_for_move`] and should be called
/// **before** reverting the bitboards.
#[inline(always)]
pub fn rollback_features_for_move(board: &mut Board, undo: &Undo, occ: &OccMap) {
    if undo.mv == PASS {
        return;
    }

    // Fast inverse of the hot-path update above.
    if board.feat_is_pattern_ids && board.feat_id_abs.len() == N_PATTERN_FEATURES {
        let flat = occ.flat();
        let mover = undo.old_side;
        let placed_delta = -(mover.digit_abs() as i32);
        update_feat_for_sq_fast(&mut board.feat_id_abs, occ, flat, undo.mv, placed_delta);

        let flip_delta: i32 = match mover {
            Color::Black => -1,
            Color::White => 1,
        };

        let mut f = undo.flips;
        while f != 0 {
            let sq = f.trailing_zeros() as u8;
            f &= f - 1;
            update_feat_for_sq_fast(&mut board.feat_id_abs, occ, flat, sq, -flip_delta);
        }
        return;
    }

    let mover = undo.old_side;
    let placed_delta = -(mover.digit_abs() as i32);
    update_feat_for_sq(board, occ, undo.mv, placed_delta);

    let flip_delta: i32 = match mover {
        Color::Black => -1,
        Color::White => 1,
    };

    let mut f = undo.flips;
    while f != 0 {
        let sq = f.trailing_zeros() as u8;
        f &= f - 1;
        update_feat_for_sq(board, occ, sq, -flip_delta);
    }
}


/// Full recomputation of all feature IDs in-place.
///
/// This is intended for the *root* initialization (Wasm boundary), not for
/// recursive search nodes. Recursive updates should use
/// [`update_features_for_move`] / [`rollback_features_for_move`].
#[inline]
pub fn recompute_features_in_place(board: &mut Board, occ: &OccMap) {
    if board.feat_id_abs.is_empty() || occ.is_empty() {
        return;
    }

    // After this function, `feat_id_abs` will contain absolute ternary pattern IDs.
    board.feat_is_pattern_ids = true;

    // Reset
    for v in board.feat_id_abs.iter_mut() {
        *v = 0;
    }

    // Accumulate digit_abs * pow3 for each occupied square.
    for sq in 0u8..64u8 {
        let bit = 1u64 << (sq as u64);
        let digit: u16 = if (board.black() & bit) != 0 {
            1
        } else if (board.white() & bit) != 0 {
            2
        } else {
            0
        };

        if digit == 0 {
            continue;
        }

        for o in occ.occ_for_sq(sq) {
            let idx = o.feature_idx as usize;
            if idx < board.feat_id_abs.len() {
                let cur = board.feat_id_abs[idx] as u32;
                let add = (digit as u32) * (o.pow3 as u32);
                let nxt = cur + add;
                board.feat_id_abs[idx] = if nxt > (u16::MAX as u32) { u16::MAX } else { nxt as u16 };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Slow but correct recompute (tests only)
// ---------------------------------------------------------------------------

#[cfg(test)]
use std::sync::OnceLock;

#[cfg(test)]
static TEST_OCC: OnceLock<OccMap> = OnceLock::new();

#[cfg(test)]
fn set_test_occ_map(occ: OccMap) {
    let _ = TEST_OCC.set(occ);
}

/// Slow but correct full recomputation of feature IDs.
///
/// **Tests only**: this relies on a test-initialized OccMap.
#[cfg(test)]
pub fn feature_id_full_recompute(board: &Board) -> Vec<u16> {
    let occ = TEST_OCC
        .get()
        .expect("TEST_OCC not initialized; call set_test_occ_map in tests");

    let mut out = vec![0u16; board.feat_id_abs.len()];

    for sq in 0u8..64u8 {
        let bit = 1u64 << (sq as u64);
        let digit: u16 = if (board.black() & bit) != 0 {
            1
        } else if (board.white() & bit) != 0 {
            2
        } else {
            0
        };

        if digit == 0 {
            continue;
        }

        for o in occ.occ_for_sq(sq) {
            let idx = o.feature_idx as usize;
            if idx < out.len() {
                let cur = out[idx] as u32;
                let add = (digit as u32) * (o.pow3 as u32);
                let nxt = cur + add;
                out[idx] = if nxt > (u16::MAX as u32) { u16::MAX } else { nxt as u16 };
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Color;

    // Deterministic RNG (SplitMix64)
    #[derive(Clone)]
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn gen_u32(&mut self, hi: u32) -> u32 {
            (self.next_u64() as u32) % hi
        }
        fn gen_bool(&mut self, num: u32, den: u32) -> bool {
            self.gen_u32(den) < num
        }
    }

    fn pick_nth_set_bit(mut mask: u64, mut n: u32) -> u8 {
        loop {
            let tz = mask.trailing_zeros();
            if tz == 64 {
                return 0;
            }
            let sq = tz as u8;
            if n == 0 {
                return sq;
            }
            n -= 1;
            mask &= mask - 1;
        }
    }

    fn popcount(x: u64) -> u32 {
        x.count_ones()
    }

    #[test]
    fn random_positions_incremental_equals_full_recompute() {
        let mut rng = Rng::new(0xC0FFEE1234567890);

        // Build a random feature set (lengths 4..=10) and its OccMap.
        let num_feats = 256usize;
        let mut feat_vecs: Vec<Vec<u8>> = Vec::with_capacity(num_feats);
        for _ in 0..num_feats {
            let len = 4 + (rng.gen_u32(7) as usize); // 4..=10
            let mut used = 0u64;
            let mut sqs = Vec::with_capacity(len);
            while sqs.len() < len {
                let sq = rng.gen_u32(64) as u8;
                let bit = 1u64 << (sq as u64);
                if (used & bit) != 0 {
                    continue;
                }
                used |= bit;
                sqs.push(sq);
            }
            feat_vecs.push(sqs);
        }
        let feat_slices: Vec<&[u8]> = feat_vecs.iter().map(|v| v.as_slice()).collect();
        let occ = OccMap::build_from_feature_squares(&feat_slices);
        set_test_occ_map(occ.clone());

        // Test multiple random boards, each with multiple random move-like edits.
        for case_idx in 0..200u32 {
            // Random board assignment: digit in {0,1,2} per square.
            let mut bbits = 0u64;
            let mut wbits = 0u64;
            for sq in 0u8..64u8 {
                match rng.gen_u32(3) {
                    1 => bbits |= 1u64 << (sq as u64),
                    2 => wbits |= 1u64 << (sq as u64),
                    _ => {}
                }
            }
            // Ensure disjoint.
            let overlap = bbits & wbits;
            bbits &= !overlap;
            wbits &= !overlap;

            let occupied = bbits | wbits;
            let empty_count = (64u32 - popcount(occupied)) as u8;

            // Keep `side=Black` for these synthetic edit tests so that
            // `player/opponent` correspond to (black/white) bitboards.
            let mut board = Board {
                player: bbits,
                opponent: wbits,
                side: Color::Black,
                empty_count,
                hash: 0,
                feat_is_pattern_ids: true,
                feat_id_abs: vec![0u16; num_feats],
            };
            board.feat_id_abs = feature_id_full_recompute(&board);

            for step in 0..50u32 {
                let mover = if rng.gen_bool(1, 2) {
                    Color::Black
                } else {
                    Color::White
                };

                let occupied = board.occupied();
                let empty = !occupied;
                if empty == 0 {
                    break;
                }

                let opp_bits = board.bits_of(mover.other());
                if opp_bits == 0 {
                    break;
                }

                // Random empty square to place.
                let n_empty = popcount(empty);
                let mv = pick_nth_set_bit(empty, rng.gen_u32(n_empty) as u32);
                let mv_bit = 1u64 << (mv as u64);

                // Random non-empty subset of opponent squares as flips (not necessarily legal Othello flips).
                let mut flips = 0u64;
                let mut tmp = opp_bits;
                while tmp != 0 {
                    let sq = tmp.trailing_zeros() as u8;
                    tmp &= tmp - 1;
                    if rng.gen_bool(1, 6) {
                        flips |= 1u64 << (sq as u64);
                    }
                }
                if flips == 0 {
                    // Force at least one flip.
                    let n_opp = popcount(opp_bits);
                    let sq = pick_nth_set_bit(opp_bits, rng.gen_u32(n_opp) as u32);
                    flips |= 1u64 << (sq as u64);
                }

                let undo = Undo {
                    mv,
                    mv_bit,
                    flips,
                    old_hash: 0,
                    old_empty: board.empty_count,
                    old_side: mover,
                };

                let before_player = board.player;
                let before_opp = board.opponent;
                let before_empty = board.empty_count;
                let before_feat = board.feat_id_abs.clone();

                // Apply bitboards.
                match mover {
                    Color::Black => {
                        board.player |= mv_bit | flips;
                        board.opponent &= !flips;
                    }
                    Color::White => {
                        // side is fixed Black, so White is stored in `opponent`.
                        board.opponent |= mv_bit | flips;
                        board.player &= !flips;
                    }
                }
                board.empty_count = board.empty_count.wrapping_sub(1);

                // Incremental update must match full recompute.
                update_features_for_move(&mut board, &undo, &occ);
                let full = feature_id_full_recompute(&board);
                assert_eq!(
                    board.feat_id_abs, full,
                    "case={case_idx} step={step} mismatch after update"
                );

                // Rollback features (still on after-move bitboards), then revert bits.
                rollback_features_for_move(&mut board, &undo, &occ);
                board.player = before_player;
                board.opponent = before_opp;
                board.empty_count = before_empty;

                let full2 = feature_id_full_recompute(&board);
                assert_eq!(
                    board.feat_id_abs, before_feat,
                    "case={case_idx} step={step} did not restore feat_id_abs"
                );
                assert_eq!(
                    board.feat_id_abs, full2,
                    "case={case_idx} step={step} mismatch after rollback"
                );
            }
        }
    }
}
