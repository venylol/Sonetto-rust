//! Move generation (legal move mask) using fixed-step Kogge–Stone expansion.
//!
//! Coordinate system: internal **bitpos** in `[0,63]` with **row-major** layout
//! (`bitpos = (row<<3)|col`).
//!
//! - `legal_moves(me, opp)` returns a bitboard of legal destination squares.
//! - The implementation is **loop-free** (fixed shifts), and matches the
//!   variable naming / structure in `Specific-details.md` (mO, f1/f7/f9/f8,
//!   pre1/pre7/pre9/pre8, ...).

use crate::coord::Move;

/// Clear file A (col == 0).
pub const NOT_A: u64 = 0xfefefefefefefefe;
/// Clear file H (col == 7).
pub const NOT_H: u64 = 0x7f7f7f7f7f7f7f7f;
/// Safe mask for non-vertical propagation (horizontal + diagonals).
pub const MASK_MO: u64 = NOT_A & NOT_H;

/// Compute legal moves for `me` against `opp`.
///
/// Returns a mask of empty squares where placing a disc flips at least one
/// opponent disc.
#[inline(always)]
pub fn legal_moves(me: u64, opp: u64) -> u64 {
    let empty = !(me | opp);
    let m_o = opp & MASK_MO;

    // forward directions
    let mut f1 = m_o & (me << 1);
    let mut f7 = m_o & (me << 7);
    let mut f9 = m_o & (me << 9);
    let mut f8 = opp & (me << 8);

    f1 |= m_o & (f1 << 1);
    f7 |= m_o & (f7 << 7);
    f9 |= m_o & (f9 << 9);
    f8 |= opp & (f8 << 8);

    let mut pre1 = m_o & (m_o << 1);
    let mut pre7 = m_o & (m_o << 7);
    let mut pre9 = m_o & (m_o << 9);
    let mut pre8 = opp & (opp << 8);

    f1 |= pre1 & (f1 << 2);
    f7 |= pre7 & (f7 << 14);
    f9 |= pre9 & (f9 << 18);
    f8 |= pre8 & (f8 << 16);

    // 再来一次覆盖到 6
    f1 |= pre1 & (f1 << 2);
    f7 |= pre7 & (f7 << 14);
    f9 |= pre9 & (f9 << 18);
    f8 |= pre8 & (f8 << 16);

    let mut moves = (f1 << 1) | (f7 << 7) | (f9 << 9) | (f8 << 8);

    // backward directions
    let mut b1 = m_o & (me >> 1);
    let mut b7 = m_o & (me >> 7);
    let mut b9 = m_o & (me >> 9);
    let mut b8 = opp & (me >> 8);

    b1 |= m_o & (b1 >> 1);
    b7 |= m_o & (b7 >> 7);
    b9 |= m_o & (b9 >> 9);
    b8 |= opp & (b8 >> 8);

    pre1 >>= 1;
    pre7 >>= 7;
    pre9 >>= 9;
    pre8 >>= 8;

    b1 |= pre1 & (b1 >> 2);
    b7 |= pre7 & (b7 >> 14);
    b9 |= pre9 & (b9 >> 18);
    b8 |= pre8 & (b8 >> 16);

    b1 |= pre1 & (b1 >> 2);
    b7 |= pre7 & (b7 >> 14);
    b9 |= pre9 & (b9 >> 18);
    b8 |= pre8 & (b8 >> 16);

    moves |= (b1 >> 1) | (b7 >> 7) | (b9 >> 9) | (b8 >> 8);

    moves & empty
}

/// Enumerate squares from a move bitmask into `out`, returning how many.
///
/// - Uses `trailing_zeros()` to extract the next set bit.
/// - Produces squares in ascending bit order.
///
/// NOTE: spec要求 `out:&mut [u8;64]`，这里用 `Move=u8` 兼容。
#[inline(always)]
pub fn push_moves_from_mask(mask: u64, out: &mut [Move; 64]) -> usize {
    let mut m = mask;
    let mut n: usize = 0;
    while m != 0 {
        let sq = m.trailing_zeros() as Move;
        out[n] = sq;
        n += 1;
        m &= m - 1;
    }
    n
}
