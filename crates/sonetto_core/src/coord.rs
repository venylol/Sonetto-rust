//! Coordinate conversions used by the engine.
//!
//! # Two coordinate systems (DO NOT MIX)
//!
//! ## External (ext packed) — UI / book / protocol
//! `ext = (col << 3) | row`, where `row,col ∈ [0,7]`.
//!
//! - `col = ext >> 3`
//! - `row = ext & 7`
//!
//! ## Internal (bitpos, row-major) — engine / bitboards
//! `bitpos = (row << 3) | col` in `[0,63]`.
//! Bitboards use `1u64 << bitpos`.
//!
//! ## Conversion
//! The conversion is **swapping the high/low 3 bits**:
//!
//! - `ext_to_bitpos(ext)  = ((ext & 7) << 3) | (ext >> 3)`
//! - `bitpos_to_ext(bit)  = ((bit & 7) << 3) | (bit >> 3)`
//!
//! Same formula in both directions (involution), so it's easy to test.

/// Engine move encoding.
/// - `0..=63` is a square in **internal bitpos**.
/// - `255` is PASS.
pub type Move = u8;

/// PASS move sentinel.
pub const PASS: Move = 0xFF;

/// Convert external `ext packed` (col<<3|row) to internal `bitpos` (row<<3|col).
#[inline(always)]
pub const fn ext_to_bitpos(ext: u8) -> u8 {
    ((ext & 7) << 3) | (ext >> 3)
}

/// Convert internal `bitpos` (row<<3|col) to external `ext packed` (col<<3|row).
#[inline(always)]
pub const fn bitpos_to_ext(bitpos: u8) -> u8 {
    ((bitpos & 7) << 3) | (bitpos >> 3)
}

/// Convert an *external* move (ext packed or PASS) to internal bitpos move (or PASS).
#[inline(always)]
pub const fn ext_move_to_bitpos_move(mv_ext: Move) -> Move {
    if mv_ext == PASS { PASS } else { ext_to_bitpos(mv_ext) }
}

/// Convert an *internal* move (bitpos or PASS) to external ext packed move (or PASS).
#[inline(always)]
pub const fn bitpos_move_to_ext_move(mv_bit: Move) -> Move {
    if mv_bit == PASS { PASS } else { bitpos_to_ext(mv_bit) }
}

/// Extract `(row, col)` from internal bitpos.
#[inline(always)]
pub const fn bitpos_to_row_col(bitpos: u8) -> (u8, u8) {
    (bitpos >> 3, bitpos & 7)
}

/// Build internal bitpos from `(row, col)`.
#[inline(always)]
pub const fn row_col_to_bitpos(row: u8, col: u8) -> u8 {
    (row << 3) | (col & 7)
}

/// Extract `(row, col)` from external ext packed.
#[inline(always)]
pub const fn ext_to_row_col(ext: u8) -> (u8, u8) {
    (ext & 7, ext >> 3)
}

/// Build external ext packed from `(row, col)`.
#[inline(always)]
pub const fn row_col_to_ext(row: u8, col: u8) -> u8 {
    ((col & 7) << 3) | (row & 7)
}

/// Internal square bit (1u64 << bitpos). Returns 0 for PASS.
#[inline(always)]
pub const fn bit_of_move(mv_bit: Move) -> u64 {
    if mv_bit == PASS { 0 } else { 1u64 << (mv_bit as u64) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_ext_bitpos_are_involutions_full_coverage() {
        for x in 0u8..=63 {
            let b = ext_to_bitpos(x);
            let x2 = bitpos_to_ext(b);
            assert_eq!(x2, x, "bitpos_to_ext(ext_to_bitpos({x})) must be {x}");
        }
        for b in 0u8..=63 {
            let x = bitpos_to_ext(b);
            let b2 = ext_to_bitpos(x);
            assert_eq!(b2, b, "ext_to_bitpos(bitpos_to_ext({b})) must be {b}");
        }
    }

    #[test]
    fn coord_row_col_roundtrip_full_coverage() {
        for bit in 0u8..=63 {
            let (r, c) = bitpos_to_row_col(bit);
            let bit2 = row_col_to_bitpos(r, c);
            assert_eq!(bit2, bit);
        }
        for row in 0u8..=7 {
            for col in 0u8..=7 {
                let ext = row_col_to_ext(row, col);
                let (r2, c2) = ext_to_row_col(ext);
                assert_eq!((r2, c2), (row, col));
            }
        }
    }

    #[test]
    fn coord_move_pass_is_preserved() {
        assert_eq!(ext_move_to_bitpos_move(PASS), PASS);
        assert_eq!(bitpos_move_to_ext_move(PASS), PASS);
        assert_eq!(bit_of_move(PASS), 0);
    }

    #[test]
    fn coord_some_known_points() {
        // ext: (col<<3)|row
        // bitpos: (row<<3)|col

        // (row=0,col=0)
        let ext_a1 = row_col_to_ext(0, 0);
        assert_eq!(ext_a1, 0);
        assert_eq!(ext_to_bitpos(ext_a1), 0);

        // (row=0,col=1) => ext=8, bitpos=1
        let ext_b1 = row_col_to_ext(0, 1);
        assert_eq!(ext_b1, 8);
        assert_eq!(ext_to_bitpos(ext_b1), 1);

        // (row=7,col=0) => ext=7, bitpos=56
        let ext_a8 = row_col_to_ext(7, 0);
        assert_eq!(ext_a8, 7);
        assert_eq!(ext_to_bitpos(ext_a8), 56);

        // (row=7,col=7) => ext=63, bitpos=63
        let ext_h8 = row_col_to_ext(7, 7);
        assert_eq!(ext_h8, 63);
        assert_eq!(ext_to_bitpos(ext_h8), 63);
    }
}
