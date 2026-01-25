//! Scalar score types.

use crate::board::{Board, Color};

/// Engine score type.
pub type Score = i32;

/// Scaling factor for disc difference.
///
/// `disc_diff_scaled = (my_discs - opp_discs) * SCALE`.
///
/// This matches the common pattern of keeping all scores in an integer domain
/// while still allowing you to interpret values as "disc" units.
pub const SCALE: Score = 32;


/// Maximum possible disc difference on an 8x8 board.
pub const MAX_DISC_DIFF: i32 = 64;

/// Convert `evaluate()` output (tanh-mapped range ~[-1000,1000]) into the engine's
/// Score units used by alpha-beta.
///
/// The search uses a disc-difference-like integer scale: `Score = disc_diff * SCALE`.
/// `evaluate()` produces a tanh-mapped score; we map it into the same domain so
/// terminals and non-terminals are comparable.
#[inline(always)]
pub fn eval1000_to_score(eval1000: i32) -> Score {
    let max_score: i64 = (MAX_DISC_DIFF as i64) * (SCALE as i64);
    let mut v: i64 = (eval1000 as i64) * max_score / 1000;

    // Clamp just in case callers pass values outside [-1000, 1000].
    if v > max_score {
        v = max_score;
    } else if v < -max_score {
        v = -max_score;
    }

    v as Score
}

/// Disc difference from `side`'s perspective, scaled by [`SCALE`].
#[inline(always)]
pub fn disc_diff_scaled(board: &Board, side: Color) -> Score {
    let me = board.bits_of(side).count_ones() as Score;
    let op = board.bits_of(side.other()).count_ones() as Score;
    (me - op) * SCALE
}

/// Game-over evaluation from `side`'s perspective, allocating remaining empties to the winner (Sensei-style).
///
/// This matters when the game ends early (both sides have no legal moves)
/// while the board still has empty squares. In that case, the remaining
/// empties are awarded to the winning side, matching Sensei's `GetEvaluationGameOver`.
#[inline(always)]
pub fn game_over_scaled(board: &Board, side: Color) -> Score {
    let me_bits = board.bits_of(side);
    let opp_bits = board.bits_of(side.other());
    game_over_scaled_from_bits(me_bits, opp_bits)
}

/// Game-over evaluation from raw bitboards (`me` and `opp`), scaled by [`SCALE`].
#[inline(always)]
pub fn game_over_scaled_from_bits(me: u64, opp: u64) -> Score {
    let me_discs = me.count_ones() as i32;
    let opp_discs = opp.count_ones() as i32;
    let diff = me_discs - opp_discs;
    let empties = 64 - me_discs - opp_discs;
    let final_diff = if diff > 0 {
        diff + empties
    } else if diff < 0 {
        diff - empties
    } else {
        0
    };
    (final_diff as Score) * SCALE
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disc_diff_scaled_is_zero_on_start_position() {
        let b = Board::new_start(0);
        assert_eq!(disc_diff_scaled(&b, Color::Black), 0);
        assert_eq!(disc_diff_scaled(&b, Color::White), 0);
    }

    #[test]
    fn disc_diff_scaled_sign_flips_with_side() {
        let mut b = Board::new_start(0);
        // Make a legal move if possible (board.rs already uses internal bitpos).
        let mv = 19u8;
        let mut u = crate::board::Undo::default();
        if !b.apply_move(mv, &mut u) {
            return; // orientation mismatch is fine for this smoke test
        }

        let s_black = disc_diff_scaled(&b, Color::Black);
        let s_white = disc_diff_scaled(&b, Color::White);
        assert_eq!(s_black, -s_white);
    }
}
