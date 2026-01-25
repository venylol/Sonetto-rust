//! Disc flipping (apply-move effect) on bitboards.
//!
//! Coordinate system: internal **bitpos** in `[0,63]` with **row-major** layout
//! (`bitpos = (row<<3)|col`). A move is represented as a **single-bit** mask
//! (`mv_bit = 1u64 << bitpos`).
//!
//! ## Fixed-step shift flips
//!
//! The baseline implementation is [`flips_for_move_shift`]. It matches the
//! structure used in `Specific-details.md`:
//!
//! - 8 directions total
//! - Horizontal + diagonals use `opp & MASK_MO` to prevent wrap-around
//! - Vertical is handled separately (no `MASK_MO`)
//!
//! ## Table-driven flips (P1-3)
//!
//! `flips_for_move_table` implements a line-table approach inspired by the
//! reference implementation (`Egaroucid/src/engine/flip_generic.hpp`).
//!
//! To keep risk low, table-driven flips are behind the `flips_table` Cargo
//! feature. The public entry point [`flips_for_move`] dispatches to either the
//! shift or the table implementation.

use std::sync::OnceLock;

/// Clear file A (col == 0).
const NOT_A: u64 = 0xfefefefefefefefe;
/// Clear file H (col == 7).
const NOT_H: u64 = 0x7f7f7f7f7f7f7f7f;
/// Safe mask for non-vertical propagation (horizontal + diagonals).
const MASK_MO: u64 = NOT_A & NOT_H;

// ---------------------------------------------------------------------------
// Shift-based flips (baseline)
// ---------------------------------------------------------------------------

#[inline(always)]
fn flips_dir_left(me: u64, opp_masked: u64, mv: u64, sh: u32) -> u64 {
    // Propagate up to 6 steps using fixed shifts (Othello max line length).
    let mut f = opp_masked & (mv << sh);
    f |= opp_masked & (f << sh);
    let pre = opp_masked & (opp_masked << sh);
    f |= pre & (f << (2 * sh));
    f |= pre & (f << (2 * sh));

    if (me & (f << sh)) != 0 { f } else { 0 }
}

#[inline(always)]
fn flips_dir_right(me: u64, opp_masked: u64, mv: u64, sh: u32) -> u64 {
    let mut f = opp_masked & (mv >> sh);
    f |= opp_masked & (f >> sh);
    let pre = opp_masked & (opp_masked >> sh);
    f |= pre & (f >> (2 * sh));
    f |= pre & (f >> (2 * sh));

    if (me & (f >> sh)) != 0 { f } else { 0 }
}

#[inline(always)]
fn flips_dir_left_v(me: u64, opp: u64, mv: u64, sh: u32) -> u64 {
    // Vertical directions do not need MASK_MO.
    let mut f = opp & (mv << sh);
    f |= opp & (f << sh);
    let pre = opp & (opp << sh);
    f |= pre & (f << (2 * sh));
    f |= pre & (f << (2 * sh));

    if (me & (f << sh)) != 0 { f } else { 0 }
}

#[inline(always)]
fn flips_dir_right_v(me: u64, opp: u64, mv: u64, sh: u32) -> u64 {
    let mut f = opp & (mv >> sh);
    f |= opp & (f >> sh);
    let pre = opp & (opp >> sh);
    f |= pre & (f >> (2 * sh));
    f |= pre & (f >> (2 * sh));

    if (me & (f >> sh)) != 0 { f } else { 0 }
}

/// Compute flipped opponent discs for a move (baseline shift implementation).
///
/// - `me`, `opp` are disjoint bitboards.
/// - `mv_bit` must be a single-bit mask.
/// - If `mv_bit` is not empty (already occupied), returns `0`.
#[inline(always)]
pub fn flips_for_move_shift(me: u64, opp: u64, mv_bit: u64) -> u64 {
    if mv_bit == 0 {
        return 0;
    }
    debug_assert!((mv_bit & (mv_bit - 1)) == 0, "mv_bit must be a single bit");
    // Spec: ensure mv_bit is empty, otherwise return 0.
    if (mv_bit & (me | opp)) != 0 {
        return 0;
    }

    let opp_mo = opp & MASK_MO;
    let mut flips = 0u64;

    // horizontal
    flips |= flips_dir_left(me, opp_mo, mv_bit, 1);
    flips |= flips_dir_right(me, opp_mo, mv_bit, 1);

    // diagonals
    flips |= flips_dir_left(me, opp_mo, mv_bit, 7);
    flips |= flips_dir_right(me, opp_mo, mv_bit, 7);

    flips |= flips_dir_left(me, opp_mo, mv_bit, 9);
    flips |= flips_dir_right(me, opp_mo, mv_bit, 9);

    // vertical
    flips |= flips_dir_left_v(me, opp, mv_bit, 8);
    flips |= flips_dir_right_v(me, opp, mv_bit, 8);

    flips
}

// ---------------------------------------------------------------------------
// Table-driven flips (P1-3)
// ---------------------------------------------------------------------------

/// Dispatch entry: use table-driven flips when `flips_table` is enabled.
///
/// This keeps the hot call sites simple (`apply`, `search`, ordering heuristics)
/// and makes it easy to A/B test.
#[inline(always)]
pub fn flips_for_move(me: u64, opp: u64, mv_bit: u64) -> u64 {
    #[cfg(feature = "flips_table")]
    {
        // Table path expects a square index.
        if mv_bit == 0 {
            return 0;
        }
        if (mv_bit & (mv_bit - 1)) != 0 {
            return 0;
        }
        let mv = mv_bit.trailing_zeros() as u8;
        flips_for_move_table(me, opp, mv)
    }

    #[cfg(not(feature = "flips_table"))]
    {
        flips_for_move_shift(me, opp, mv_bit)
    }
}

/// Compute flipped discs for a move, assuming the caller already validated:
/// - `mv_bit` is a single bit (power-of-two)
/// - `mv_bit` is empty (does not overlap `me|opp`)
///
/// This is a hot-path helper used by the recursive search / make-move code to
/// avoid redundant legality checks.
#[inline(always)]
pub fn flips_for_move_unchecked(me: u64, opp: u64, mv_bit: u64) -> u64 {
    debug_assert!(mv_bit != 0);
    debug_assert!((mv_bit & (mv_bit - 1)) == 0);
    debug_assert!((mv_bit & (me | opp)) == 0);

    #[cfg(feature = "flips_table")]
    {
        let mv = mv_bit.trailing_zeros() as u8;
        return flips_for_move_table_unchecked(me, opp, mv);
    }

    #[cfg(not(feature = "flips_table"))]
    {
        flips_for_move_shift_unchecked(me, opp, mv_bit)
    }
}

#[inline(always)]
fn flips_for_move_shift_unchecked(me: u64, opp: u64, mv_bit: u64) -> u64 {
    let opp_mo = opp & MASK_MO;
    let mut flips = 0u64;

    // horizontal
    flips |= flips_dir_left(me, opp_mo, mv_bit, 1);
    flips |= flips_dir_right(me, opp_mo, mv_bit, 1);

    // diagonals
    flips |= flips_dir_left(me, opp_mo, mv_bit, 7);
    flips |= flips_dir_right(me, opp_mo, mv_bit, 7);

    flips |= flips_dir_left(me, opp_mo, mv_bit, 9);
    flips |= flips_dir_right(me, opp_mo, mv_bit, 9);

    // vertical
    flips |= flips_dir_left_v(me, opp, mv_bit, 8);
    flips |= flips_dir_right_v(me, opp, mv_bit, 8);

    flips
}

const FLIP8_TABLE_SIZE: usize = 256 * 256 * 8;

/// FLIP8[pos][me8][opp8] -> flips8 (8-bit line representation).
///
/// We store a single flat array for cache friendliness.
static FLIP8_LUT: OnceLock<Box<[u8]>> = OnceLock::new();

#[inline(always)]
fn flip8_table() -> &'static [u8] {
    FLIP8_LUT.get_or_init(|| {
        let mut t = vec![0u8; FLIP8_TABLE_SIZE];

        // Fill all combinations (including invalid overlaps => 0).
        for me in 0u16..=255 {
            for opp in 0u16..=255 {
                let me8 = me as u8;
                let opp8 = opp as u8;
                if (me8 & opp8) != 0 {
                    continue;
                }
                for pos in 0u8..8 {
                    let idx = ((me as usize) << 11) | ((opp as usize) << 3) | (pos as usize);
                    t[idx] = calc_flip8(me8, opp8, pos);
                }
            }
        }

        t.into_boxed_slice()
    })
}

/// P0-3: Warm up the flip line table.
///
/// This is only relevant when the `flips_table` feature is enabled.
#[cfg(feature = "flips_table")]
#[inline]
pub fn warm_up_flip_tables() {
    let _ = flip8_table();
}

#[inline(always)]
fn flip8(me8: u8, opp8: u8, pos: u8) -> u8 {
    debug_assert!(pos < 8);

    // idx = me*256*8 + opp*8 + pos
    let idx = ((me8 as usize) << 11) | ((opp8 as usize) << 3) | (pos as usize);
    flip8_table()[idx]
}

#[inline(always)]
fn calc_flip8(me8: u8, opp8: u8, pos: u8) -> u8 {
    let occ = me8 | opp8;
    let mv_bit = 1u8 << pos;
    if (occ & mv_bit) != 0 {
        return 0;
    }

    let mut flips = 0u8;

    // Right direction (increasing index).
    let mut i = pos + 1;
    let mut acc = 0u8;
    while i < 8 {
        let b = 1u8 << i;
        if (opp8 & b) != 0 {
            acc |= b;
            i += 1;
            continue;
        }
        if (me8 & b) != 0 {
            flips |= acc;
        }
        break;
    }

    // Left direction (decreasing index).
    let mut j: i8 = (pos as i8) - 1;
    let mut acc2 = 0u8;
    while j >= 0 {
        let b = 1u8 << (j as u8);
        if (opp8 & b) != 0 {
            acc2 |= b;
            j -= 1;
            continue;
        }
        if (me8 & b) != 0 {
            flips |= acc2;
        }
        break;
    }

    flips
}

// Column extraction is based on the classic multiply+shift trick.
// (Same as the reference implementation.)
#[inline(always)]
fn extract_col8(bb: u64, col: u8) -> u8 {
    debug_assert!(col < 8);
    let x = (bb >> (col as u64)) & 0x0101_0101_0101_0101u64;
    ((x.wrapping_mul(0x0102_0408_1020_4080u64)) >> 56) as u8
}

// Precomputed: 8-bit column mask -> u64 with bits at (row*8+0).
const COL8_TO_BB: [u64; 256] = build_col8_to_bb();

const fn build_col8_to_bb() -> [u64; 256] {
    let mut t = [0u64; 256];
    let mut v: usize = 0;
    while v < 256 {
        let mut out = 0u64;
        let mut r: usize = 0;
        while r < 8 {
            if (v & (1usize << r)) != 0 {
                out |= 1u64 << (r * 8);
            }
            r += 1;
        }
        t[v] = out;
        v += 1;
    }
    t
}

// ---------------------------------------------------------------------------
// Diagonal line extraction/scatter helpers (WASM-friendly)
// ---------------------------------------------------------------------------
//
// Motivation:
// - On WASM, very short `while` loops with branches are expensive.
// - Diagonal/anti-diagonal extraction in table-driven flips used to iterate
//   `len` (1..8) squares with per-iteration branching.
//
// This implementation precomputes, for every move square:
// - the 8 squares of its diagonal line as bit masks (unused slots are 0)
// - the move's position in that 8-bit line (`pos`)
// - a `valid_mask` to clear unused bits when the line length < 8
//
// Then we implement extract/scatter as 8 fixed operations (no loop), keeping
// the hot flip path branchless and JIT-friendly.

#[derive(Copy, Clone)]
struct LineInfo {
    /// mv in this 8-bit line: [0..7].
    pos: u8,
    /// Valid bits for this line: len==8 => 0xFF, else (1<<len)-1.
    valid_mask: u8,
    /// Bitboard masks for each line slot. For i>=len this is 0.
    masks: [u64; 8],
}

impl LineInfo {
    const ZERO: LineInfo = LineInfo {
        pos: 0,
        valid_mask: 0,
        masks: [0u64; 8],
    };
}

#[inline(always)]
const fn u8_min(a: u8, b: u8) -> u8 {
    if a < b { a } else { b }
}

const fn build_diag9() -> [LineInfo; 64] {
    let mut out = [LineInfo::ZERO; 64];
    let mut mv: u8 = 0;
    while mv < 64 {
        let r = mv >> 3;
        let c = mv & 7;

        // Position within the diagonal is the distance to the NW end.
        let k = if r < c { r } else { c };
        let sr = r - k;
        let sc = c - k;

        // Diagonal length is bounded by board edges.
        let len = u8_min(8 - sr, 8 - sc); // 1..8
        let start = sr * 8 + sc;

        let valid_mask = if len == 8 {
            0xFF
        } else {
            ((1u16 << len) - 1) as u8
        };

        let mut masks = [0u64; 8];
        let mut i: u8 = 0;
        while i < 8 {
            if i < len {
                let sq = start + i * 9; // step=9
                masks[i as usize] = 1u64 << (sq as u64);
            }
            i += 1;
        }

        out[mv as usize] = LineInfo {
            pos: k,
            valid_mask,
            masks,
        };
        mv += 1;
    }
    out
}

const fn build_diag7() -> [LineInfo; 64] {
    let mut out = [LineInfo::ZERO; 64];
    let mut mv: u8 = 0;
    while mv < 64 {
        let r = mv >> 3;
        let c = mv & 7;

        // Anti-diagonal position is distance to the NE end.
        let t = 7 - c;
        let k = if r < t { r } else { t };
        let sr = r - k;
        let sc = c + k;

        let len = u8_min(8 - sr, sc + 1); // 1..8
        let start = sr * 8 + sc;

        let valid_mask = if len == 8 {
            0xFF
        } else {
            ((1u16 << len) - 1) as u8
        };

        let mut masks = [0u64; 8];
        let mut i: u8 = 0;
        while i < 8 {
            if i < len {
                let sq = start + i * 7; // step=7
                masks[i as usize] = 1u64 << (sq as u64);
            }
            i += 1;
        }

        out[mv as usize] = LineInfo {
            pos: k,
            valid_mask,
            masks,
        };
        mv += 1;
    }
    out
}

const DIAG9: [LineInfo; 64] = build_diag9();
const DIAG7: [LineInfo; 64] = build_diag7();

#[inline(always)]
fn extract_line_bits_unrolled(me: u64, opp: u64, info: &LineInfo) -> (u8, u8) {
    let m = &info.masks;

    let me8: u8 =
        (((me & m[0]) != 0) as u8) << 0 |
        (((me & m[1]) != 0) as u8) << 1 |
        (((me & m[2]) != 0) as u8) << 2 |
        (((me & m[3]) != 0) as u8) << 3 |
        (((me & m[4]) != 0) as u8) << 4 |
        (((me & m[5]) != 0) as u8) << 5 |
        (((me & m[6]) != 0) as u8) << 6 |
        (((me & m[7]) != 0) as u8) << 7;

    let opp8: u8 =
        (((opp & m[0]) != 0) as u8) << 0 |
        (((opp & m[1]) != 0) as u8) << 1 |
        (((opp & m[2]) != 0) as u8) << 2 |
        (((opp & m[3]) != 0) as u8) << 3 |
        (((opp & m[4]) != 0) as u8) << 4 |
        (((opp & m[5]) != 0) as u8) << 5 |
        (((opp & m[6]) != 0) as u8) << 6 |
        (((opp & m[7]) != 0) as u8) << 7;

    (me8, opp8)
}

#[inline(always)]
fn scatter_line_bits_unrolled(bits8: u8, info: &LineInfo) -> u64 {
    let m = &info.masks;

    // sel = 0xffff.. if bit==1 else 0
    let s0 = 0u64.wrapping_sub(((bits8 >> 0) & 1) as u64);
    let s1 = 0u64.wrapping_sub(((bits8 >> 1) & 1) as u64);
    let s2 = 0u64.wrapping_sub(((bits8 >> 2) & 1) as u64);
    let s3 = 0u64.wrapping_sub(((bits8 >> 3) & 1) as u64);
    let s4 = 0u64.wrapping_sub(((bits8 >> 4) & 1) as u64);
    let s5 = 0u64.wrapping_sub(((bits8 >> 5) & 1) as u64);
    let s6 = 0u64.wrapping_sub(((bits8 >> 6) & 1) as u64);
    let s7 = 0u64.wrapping_sub(((bits8 >> 7) & 1) as u64);

    (m[0] & s0)
        | (m[1] & s1)
        | (m[2] & s2)
        | (m[3] & s3)
        | (m[4] & s4)
        | (m[5] & s5)
        | (m[6] & s6)
        | (m[7] & s7)
}

/// Table-based flip implementation (line-table driven).
///
/// - `mv` is a square index in `[0,63]`.
/// - Returns `0` for illegal/occupied moves (same contract as shift version).
#[inline(always)]
pub fn flips_for_move_table(me: u64, opp: u64, mv: u8) -> u64 {
    if mv > 63 {
        return 0;
    }

    let mv_bit = 1u64 << (mv as u64);
    if (mv_bit & (me | opp)) != 0 {
        return 0;
    }

    // P0-6: BMI2 pext/pdep accelerated path (Egaroucid-style)
    #[cfg(all(target_arch = "x86_64", feature = "flips_table"))]
    {
        if std::is_x86_feature_detected!("bmi2") {
            // Safety: guarded by runtime CPU feature detection.
            unsafe { return flips_bmi2::flips_for_move_table_bmi2(me, opp, mv); }
        }
    }


    let row = mv >> 3;
    let col = mv & 7;

    // --- Row (E/W) ---
    let row_shift = (row as u64) * 8;
    let me_row8 = ((me >> row_shift) & 0xFF) as u8;
    let opp_row8 = ((opp >> row_shift) & 0xFF) as u8;
    let flips_row8 = flip8(me_row8, opp_row8, col);
    let flips_row = (flips_row8 as u64) << row_shift;

    // --- Column (N/S) ---
    let me_col8 = extract_col8(me, col);
    let opp_col8 = extract_col8(opp, col);
    let flips_col8 = flip8(me_col8, opp_col8, row);
    let flips_col = COL8_TO_BB[flips_col8 as usize] << (col as u64);

    // --- Diagonal (NW/SE, step=9) ---
    let info9 = &DIAG9[mv as usize];
    let (me_d9, opp_d9) = extract_line_bits_unrolled(me, opp, info9);
    let flips_d9_8 = flip8(me_d9, opp_d9, info9.pos) & info9.valid_mask;
    let flips_d9 = scatter_line_bits_unrolled(flips_d9_8, info9);

    // --- Anti-diagonal (NE/SW, step=7) ---
    let info7 = &DIAG7[mv as usize];
    let (me_d7, opp_d7) = extract_line_bits_unrolled(me, opp, info7);
    let flips_d7_8 = flip8(me_d7, opp_d7, info7.pos) & info7.valid_mask;
    let flips_d7 = scatter_line_bits_unrolled(flips_d7_8, info7);

    flips_row | flips_col | flips_d9 | flips_d7
}

/// Table-based flip implementation without redundancy checks.
///
/// Preconditions (debug-asserted):
/// - `mv` in `[0,63]`
/// - `mv` is empty (does not overlap `me|opp`)
#[inline(always)]
fn flips_for_move_table_unchecked(me: u64, opp: u64, mv: u8) -> u64 {
    debug_assert!(mv < 64);
    let mv_bit = 1u64 << (mv as u64);
    debug_assert!((mv_bit & (me | opp)) == 0);

    // P0-6: BMI2 pext/pdep accelerated path (Egaroucid-style)
    #[cfg(all(target_arch = "x86_64", feature = "flips_table"))]
    {
        if std::is_x86_feature_detected!("bmi2") {
            // Safety: guarded by runtime CPU feature detection.
            unsafe { return flips_bmi2::flips_for_move_table_bmi2(me, opp, mv); }
        }
    }


    let row = mv >> 3;
    let col = mv & 7;

    // --- Row (E/W) ---
    let row_shift = (row as u64) * 8;
    let me_row8 = ((me >> row_shift) & 0xFF) as u8;
    let opp_row8 = ((opp >> row_shift) & 0xFF) as u8;
    let flips_row8 = flip8(me_row8, opp_row8, col);
    let flips_row = (flips_row8 as u64) << row_shift;

    // --- Column (N/S) ---
    let me_col8 = extract_col8(me, col);
    let opp_col8 = extract_col8(opp, col);
    let flips_col8 = flip8(me_col8, opp_col8, row);
    let flips_col = COL8_TO_BB[flips_col8 as usize] << (col as u64);

    // --- Diagonal (NW/SE, step=9) ---
    let info9 = &DIAG9[mv as usize];
    let (me_d9, opp_d9) = extract_line_bits_unrolled(me, opp, info9);
    let flips_d9_8 = flip8(me_d9, opp_d9, info9.pos) & info9.valid_mask;
    let flips_d9 = scatter_line_bits_unrolled(flips_d9_8, info9);

    // --- Anti-diagonal (NE/SW, step=7) ---
    let info7 = &DIAG7[mv as usize];
    let (me_d7, opp_d7) = extract_line_bits_unrolled(me, opp, info7);
    let flips_d7_8 = flip8(me_d7, opp_d7, info7.pos) & info7.valid_mask;
    let flips_d7 = scatter_line_bits_unrolled(flips_d7_8, info7);

    flips_row | flips_col | flips_d9 | flips_d7
}



// ---------------------------------------------------------------------------
// BMI2 accelerated table-driven flips (P0-6)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "flips_table", target_arch = "x86_64"))]
mod flips_bmi2 {
    use core::arch::x86_64::{_pdep_u64, _pext_u64};

    #[inline(always)]
    const fn u8_min(a: u8, b: u8) -> u8 {
        if a < b { a } else { b }
    }

    #[inline(always)]
    const fn diag9_mask_for_sq(sq: u8) -> u64 {
        let row = (sq >> 3) as i32;
        let col = (sq & 7) as i32;
        let k = if row < col { row } else { col };

        let mut r = row - k;
        let mut c = col - k;
        let mut mask: u64 = 0;
        while r < 8 && c < 8 {
            let idx = (r * 8 + c) as u64;
            mask |= 1u64 << idx;
            r += 1;
            c += 1;
        }
        mask
    }

    #[inline(always)]
    const fn diag7_mask_for_sq(sq: u8) -> u64 {
        let row = (sq >> 3) as i32;
        let col = (sq & 7) as i32;
        let inv_col = 7 - col;
        let k = if row < inv_col { row } else { inv_col };

        let mut r = row - k;
        let mut c = col + k;
        let mut mask: u64 = 0;
        while r < 8 && c >= 0 {
            let idx = (r * 8 + c) as u64;
            mask |= 1u64 << idx;
            r += 1;
            c -= 1;
        }
        mask
    }

    #[inline(always)]
    const fn build_diag9_masks() -> [u64; 64] {
        let mut out = [0u64; 64];
        let mut i = 0usize;
        while i < 64 {
            out[i] = diag9_mask_for_sq(i as u8);
            i += 1;
        }
        out
    }

    #[inline(always)]
    const fn build_diag7_masks() -> [u64; 64] {
        let mut out = [0u64; 64];
        let mut i = 0usize;
        while i < 64 {
            out[i] = diag7_mask_for_sq(i as u8);
            i += 1;
        }
        out
    }

    pub const DIAG9_MASK: [u64; 64] = build_diag9_masks();
    pub const DIAG7_MASK: [u64; 64] = build_diag7_masks();

    pub const COL_MASK: [u64; 8] = [
        0x0101_0101_0101_0101u64 << 0,
        0x0101_0101_0101_0101u64 << 1,
        0x0101_0101_0101_0101u64 << 2,
        0x0101_0101_0101_0101u64 << 3,
        0x0101_0101_0101_0101u64 << 4,
        0x0101_0101_0101_0101u64 << 5,
        0x0101_0101_0101_0101u64 << 6,
        0x0101_0101_0101_0101u64 << 7,
    ];

    /// BMI2 accelerated flips for a **known-empty** move square.
    ///
    /// # Safety
    /// - The caller must ensure `mv` is in `[0,63]` and the square is empty.
    /// - The caller must ensure the CPU supports BMI2 (runtime detection).
    #[inline(always)]
    #[target_feature(enable = "bmi2")]
    pub unsafe fn flips_for_move_table_bmi2(me: u64, opp: u64, mv: u8) -> u64 {
        debug_assert!(mv < 64);
        let mv_bit = 1u64 << (mv as u64);
        debug_assert!((mv_bit & (me | opp)) == 0);

        let row = mv >> 3;
        let col = mv & 7;

        // --- Row (E/W) --- (shift is already constant-time)
        let row_shift = (row as u64) * 8;
        let me_row8 = ((me >> row_shift) & 0xFF) as u8;
        let opp_row8 = ((opp >> row_shift) & 0xFF) as u8;
        let flips_row8 = super::flip8(me_row8, opp_row8, col);
        let flips_row = (flips_row8 as u64) << row_shift;

        // --- Column (N/S): pext/pdep ---
        let cm = COL_MASK[col as usize];
        let me_col8 = _pext_u64(me, cm) as u8;
        let opp_col8 = _pext_u64(opp, cm) as u8;
        let flips_col8 = super::flip8(me_col8, opp_col8, row);
        let flips_col = _pdep_u64(flips_col8 as u64, cm);

        // --- Diagonal (NW/SE, step=9): pext/pdep ---
        let d9m = DIAG9_MASK[mv as usize];
        let me_d9 = _pext_u64(me, d9m) as u8;
        let opp_d9 = _pext_u64(opp, d9m) as u8;
        let pos_d9 = u8_min(row, col);

        // Length = 8 - |row-col|
        let diff = if row > col { row - col } else { col - row };
        let d9_len = 8u8.saturating_sub(diff);

        let mut flips_d9_8 = super::flip8(me_d9, opp_d9, pos_d9);
        if d9_len < 8 {
            flips_d9_8 &= (1u8 << d9_len).wrapping_sub(1);
        }
        let flips_d9 = _pdep_u64(flips_d9_8 as u64, d9m);

        // --- Anti-diagonal (NE/SW, step=7): pext/pdep ---
        let d7m = DIAG7_MASK[mv as usize];
        let me_d7 = _pext_u64(me, d7m) as u8;
        let opp_d7 = _pext_u64(opp, d7m) as u8;
        let inv_col = 7u8 - col;
        let pos_d7 = u8_min(row, inv_col);

        // Length = 8 - |(row+col) - 7|
        let sum = row + col;
        let diff2 = if sum > 7 { sum - 7 } else { 7 - sum };
        let d7_len = 8u8.saturating_sub(diff2);

        let mut flips_d7_8 = super::flip8(me_d7, opp_d7, pos_d7);
        if d7_len < 8 {
            flips_d7_8 &= (1u8 << d7_len).wrapping_sub(1);
        }
        let flips_d7 = _pdep_u64(flips_d7_8 as u64, d7m);

        flips_row | flips_col | flips_d9 | flips_d7
    }
}
// --------------------
// Tests
// --------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[inline(always)]
    fn bit(sq: u8) -> u64 {
        1u64 << (sq as u64)
    }

    /// Naive 8-direction while-loop implementation for correctness testing.
    fn flips_for_move_naive(me: u64, opp: u64, mv_bit: u64) -> u64 {
        if mv_bit == 0 {
            return 0;
        }
        if (mv_bit & (me | opp)) != 0 {
            return 0;
        }

        let sq = mv_bit.trailing_zeros() as i32;
        let r0 = sq / 8;
        let c0 = sq % 8;

        const DIRS: [(i32, i32); 8] = [
            (-1, -1), // NW
            (-1, 0),  // N
            (-1, 1),  // NE
            (0, -1),  // W
            (0, 1),   // E
            (1, -1),  // SW
            (1, 0),   // S
            (1, 1),   // SE
        ];

        let mut flips = 0u64;
        for (dr, dc) in DIRS {
            let mut r = r0 + dr;
            let mut c = c0 + dc;
            let mut acc = 0u64;
            while (0..=7).contains(&r) && (0..=7).contains(&c) {
                let sq2 = (r * 8 + c) as u8;
                let b = bit(sq2);
                if (opp & b) != 0 {
                    acc |= b;
                    r += dr;
                    c += dc;
                    continue;
                }
                if (me & b) != 0 {
                    flips |= acc;
                }
                break; // reached me or empty
            }
        }
        flips
    }

    /// Small, deterministic PRNG (SplitMix64).
    #[derive(Clone)]
    struct SplitMix64 {
        x: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { x: seed }
        }

        #[inline(always)]
        fn next_u64(&mut self) -> u64 {
            self.x = self.x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        #[inline(always)]
        fn next_u8(&mut self) -> u8 {
            (self.next_u64() & 0xFF) as u8
        }
    }

    #[test]
    fn flips_shift_matches_naive_random_positions_and_moves() {
        let mut rng = SplitMix64::new(0x0123_4567_89AB_CDEF);

        // Random, not-necessarily-reachable positions are fine: we only test flip logic.
        for _ in 0..20_000 {
            let me = rng.next_u64();
            let opp = rng.next_u64() & !me; // ensure disjoint

            // random move (often occupied, which should yield 0 in in both)
            let mv = rng.next_u8() & 63;
            let mv_bit = bit(mv);

            let f_shift = flips_for_move_shift(me, opp, mv_bit);
            let f_naive = flips_for_move_naive(me, opp, mv_bit);

            assert_eq!(f_shift, f_naive, "mismatch: mv={mv}, me={me:#018x}, opp={opp:#018x}");
        }
    }

    #[test]
    fn flips_shift_matches_naive_on_all_moves_for_some_random_positions() {
        let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_BABE);
        for _ in 0..256 {
            let me = rng.next_u64();
            let opp = rng.next_u64() & !me;
            for mv in 0u8..64 {
                let mv_bit = bit(mv);
                let f_shift = flips_for_move_shift(me, opp, mv_bit);
                let f_naive = flips_for_move_naive(me, opp, mv_bit);
                assert_eq!(f_shift, f_naive, "mismatch: mv={mv}, me={me:#018x}, opp={opp:#018x}");
            }
        }
    }

    #[test]
    fn flips_table_matches_shift_random_positions_and_moves() {
        let mut rng = SplitMix64::new(0x1357_9BDF_2468_ACE0);
        for _ in 0..20_000 {
            let me = rng.next_u64();
            let opp = rng.next_u64() & !me;

            let mv = rng.next_u8() & 63;
            let mv_bit = bit(mv);

            let f_shift = flips_for_move_shift(me, opp, mv_bit);
            let f_table = flips_for_move_table(me, opp, mv);

            assert_eq!(f_shift, f_table, "table mismatch: mv={mv}, me={me:#018x}, opp={opp:#018x}");
        }
    }

    #[test]
    fn flips_table_matches_naive_on_all_moves_for_some_random_positions() {
        let mut rng = SplitMix64::new(0x0BAD_F00D_DEAD_BEEF);
        for _ in 0..128 {
            let me = rng.next_u64();
            let opp = rng.next_u64() & !me;
            for mv in 0u8..64 {
                let mv_bit = bit(mv);
                let f_table = flips_for_move_table(me, opp, mv);
                let f_naive = flips_for_move_naive(me, opp, mv_bit);
                assert_eq!(f_table, f_naive, "mismatch: mv={mv}, me={me:#018x}, opp={opp:#018x}");
            }
        }
    }

    #[test]
    fn col8_to_bb_roundtrip_matches_extract_col8() {
        // Simple random check: expand(extract(bb,col)) should equal bb & column_mask.
        let mut rng = SplitMix64::new(0xCAFEBABE_0123_4567);
        for _ in 0..4096 {
            let bb = rng.next_u64();
            for col in 0u8..8 {
                let x = extract_col8(bb, col);
                let expanded = COL8_TO_BB[x as usize] << (col as u64);
                let col_mask = 0x0101_0101_0101_0101u64 << (col as u64);
                assert_eq!(expanded, bb & col_mask);
            }
        }
    }
}
