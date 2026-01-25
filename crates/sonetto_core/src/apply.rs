//! Compatibility wrappers for make/undo.
//!
//! Historically Sonetto exposed free functions in this module.
//! The canonical implementation now lives on [`Board`](crate::board::Board)
//! (see `board.rs`, which implements P2-3 player/opponent swapping).
//!
//! Keeping these thin wrappers minimizes API churn and ensures all call sites
//! share exactly the same make/undo semantics.

use crate::board::{Board, Color, Undo};
use crate::coord::{Move, PASS};
use crate::features::occ::OccMap;

/// Apply a move (`0..63` internal bitpos) or [`PASS`].
#[inline(always)]
pub fn apply_move(board: &mut Board, mv: Move, undo: &mut Undo) -> bool {
    board.apply_move(mv, undo)
}

/// Apply a move, optionally with an [`OccMap`] to incrementally update pattern IDs.
#[inline(always)]
pub fn apply_move_with_occ(board: &mut Board, mv: Move, undo: &mut Undo, occ: Option<&OccMap>) -> bool {
    board.apply_move_with_occ(mv, undo, occ)
}

/// Undo a move previously applied by [`apply_move`].
#[inline(always)]
pub fn undo_move(board: &mut Board, undo: &Undo) {
    board.undo_move(undo)
}

/// Undo a move previously applied by [`apply_move_with_occ`].
#[inline(always)]
pub fn undo_move_with_occ(board: &mut Board, undo: &Undo, occ: Option<&OccMap>) {
    board.undo_move_with_occ(undo, occ)
}

// ---------------------------------------------------------------------------
// Legacy per-square digit cache helpers
// ---------------------------------------------------------------------------
//
// These helpers are preserved for compatibility with earlier phases where
// `feat_id_abs.len()==64` was used as a per-square abs digit cache.
//
// The engine's mainline now uses `features::update` + OccMap incremental updates
// for pattern IDs, but external tooling/tests may still rely on these.

/// Apply feature changes for a move when `feat_id_abs` is used as per-square digits.
#[inline(always)]
pub fn update_sq_features_for_move(board: &mut Board, undo: &Undo) {
    if undo.mv == PASS {
        return;
    }
    if board.feat_id_abs.len() != 64 {
        return;
    }

    let mover = undo.old_side;
    let digit = mover.digit_abs();

    // placed: empty -> mover
    board.feat_id_abs[undo.mv as usize] = digit;

    // flips: opp -> mover
    let mut f = undo.flips;
    while f != 0 {
        let sq = f.trailing_zeros() as usize;
        f &= f - 1;
        board.feat_id_abs[sq] = digit;
    }
}

/// Roll back feature changes for a move when `feat_id_abs` is used as per-square digits.
#[inline(always)]
pub fn rollback_sq_features_for_move(board: &mut Board, undo: &Undo) {
    if undo.mv == PASS {
        return;
    }
    if board.feat_id_abs.len() != 64 {
        return;
    }

    let mover = undo.old_side;
    let opp_digit = mover.other().digit_abs();

    // placed: mover -> empty
    board.feat_id_abs[undo.mv as usize] = 0;

    // flips: mover -> opp
    let mut f = undo.flips;
    while f != 0 {
        let sq = f.trailing_zeros() as usize;
        f &= f - 1;
        board.feat_id_abs[sq] = opp_digit;
    }
}

/// Recompute the per-square digit cache from the bitboards.
///
/// This is intentionally small and safe for tooling.
#[inline]
pub fn recompute_sq_features_from_bitboards(board: &mut Board) {
    if board.feat_id_abs.len() != 64 {
        return;
    }

    board.feat_id_abs.fill(0);

    let black = board.bits_of(Color::Black);
    let white = board.bits_of(Color::White);

    let mut b = black;
    while b != 0 {
        let sq = b.trailing_zeros() as usize;
        b &= b - 1;
        board.feat_id_abs[sq] = Color::Black.digit_abs();
    }
    let mut w = white;
    while w != 0 {
        let sq = w.trailing_zeros() as usize;
        w &= w - 1;
        board.feat_id_abs[sq] = Color::White.digit_abs();
    }
}
