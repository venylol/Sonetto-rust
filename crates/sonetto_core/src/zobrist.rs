//! Deterministic Zobrist-style hashing.
//!
//! Hash update logic matches Sonetto's incremental scheme:
//!
//! - **place**: `hash ^= piece_key(mover, mv)`
//! - **flips**: for each flipped square `sq`,
//!   `hash ^= piece_key(opp, sq) ^ piece_key(mover, sq)`
//! - **side**: after changing side-to-move, `hash ^= side_key()`
//!
//! Phase0 P0-1: precompute all keys in a small `[2][64]` table so hot paths
//! don't run the SplitMix64 mixer per key lookup.

use std::sync::OnceLock;

use crate::board::Color;

/// Fixed seed for key derivation.
pub const ZOBRIST_SEED: u64 = 0x6A09_E667_F3BC_C909;

#[inline(always)]
fn mix64(mut x: u64) -> u64 {
    // SplitMix64 finalizer.
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[derive(Clone, Copy)]
struct ZobristTables {
    piece_keys: [[u64; 64]; 2],
    side_key: u64,
    /// Byte-wise XOR table for fast hashing.
    ///
    /// `xor8[color][byte_index][byte_value]` is the XOR of `piece_keys[color][sq]`
    /// for all set bits in `byte_value`, where `sq = byte_index*8 + bit`.
    ///
    /// This allows computing `xor_piece_keys(color, mask)` in a fixed 8 table
    /// lookups (no per-bit loops), which significantly speeds up:
    /// - full hash recomputation at the root / WASM boundary
    /// - incremental updates for flips in make/undo
    xor8: [[[u64; 256]; 8]; 2],
}

// Runtime-initialized tables.
static TABLES: OnceLock<ZobristTables> = OnceLock::new();

#[inline(always)]
fn tables() -> &'static ZobristTables {
    TABLES.get_or_init(|| {
        let mut piece_keys = [[0u64; 64]; 2];
        let mut color: usize = 0;
        while color < 2 {
            let mut sq: usize = 0;
            while sq < 64 {
                // unique id in [0,127]
                let id = (color as u64) * 64 + (sq as u64);
                piece_keys[color][sq] = mix64(ZOBRIST_SEED ^ id);
                sq += 1;
            }
            color += 1;
        }

        // Convention: `side_key()` is XOR-ed in **iff** `side == White`.
        // This makes `hash ^= side_key()` a simple toggle on side changes.
        let side_key = mix64(ZOBRIST_SEED ^ 0xFFFF_FFFF_FFFF_FFFF);

        // Build byte-wise XOR table.
        // 2 colors * 8 bytes * 256 values = 4096 entries (~32KB).
        let mut xor8 = [[[0u64; 256]; 8]; 2];
        let mut c: usize = 0;
        while c < 2 {
            let mut byte_idx: usize = 0;
            while byte_idx < 8 {
                let base_sq = byte_idx * 8;
                let mut v: usize = 0;
                while v < 256 {
                    let mut x: u64 = 0;
                    let mut bit: usize = 0;
                    while bit < 8 {
                        if (v & (1usize << bit)) != 0 {
                            x ^= piece_keys[c][base_sq + bit];
                        }
                        bit += 1;
                    }
                    xor8[c][byte_idx][v] = x;
                    v += 1;
                }
                byte_idx += 1;
            }
            c += 1;
        }

        ZobristTables { piece_keys, side_key, xor8 }
    })
}

/// XOR of `piece_key(color, sq)` for all set squares in `mask`.
///
/// This is a hot primitive used both for full hash recomputation and for
/// incremental make/undo updates (flips).
#[inline(always)]
pub fn xor_piece_keys(color: Color, mask: u64) -> u64 {
    let t = tables();
    let c = color.idx();

    // Least-significant byte corresponds to squares 0..7, then 8..15, etc.
    let b = mask.to_le_bytes();

    // Manual unroll: faster than a loop in some backends (especially WASM).
    t.xor8[c][0][b[0] as usize]
        ^ t.xor8[c][1][b[1] as usize]
        ^ t.xor8[c][2][b[2] as usize]
        ^ t.xor8[c][3][b[3] as usize]
        ^ t.xor8[c][4][b[4] as usize]
        ^ t.xor8[c][5][b[5] as usize]
        ^ t.xor8[c][6][b[6] as usize]
        ^ t.xor8[c][7][b[7] as usize]
}

/// Per-square piece key.
#[inline(always)]
pub fn piece_key(color: Color, sq: u8) -> u64 {
    debug_assert!(sq < 64);
    tables().piece_keys[color.idx()][sq as usize]
}

/// Side-to-move toggle key.
#[inline(always)]
pub fn side_key() -> u64 {
    tables().side_key
}

/// Compute a full hash from scratch.
///
/// Convention: `side_key()` is XOR-ed in **iff** `side == White`.
#[inline]
pub fn compute_hash(bits: [u64; 2], side: Color) -> u64 {
    let mut h = xor_piece_keys(Color::Black, bits[Color::Black.idx()])
        ^ xor_piece_keys(Color::White, bits[Color::White.idx()]);

    if side == Color::White {
        h ^= side_key();
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_key_is_a_toggle() {
        let bits = [0u64, 0u64];
        let h_b = compute_hash(bits, Color::Black);
        let h_w = compute_hash(bits, Color::White);
        assert_eq!(h_b ^ side_key(), h_w);
    }

    #[test]
    fn table_matches_mixer_definition() {
        // Guard against accidental table drift.
        for color_idx in 0..2usize {
            for sq in 0..64usize {
                let id = (color_idx as u64) * 64 + (sq as u64);
                let expected = mix64(ZOBRIST_SEED ^ id);
                let c = if color_idx == 0 { Color::Black } else { Color::White };
                assert_eq!(piece_key(c, sq as u8), expected);
            }
        }
        assert_eq!(side_key(), mix64(ZOBRIST_SEED ^ 0xFFFF_FFFF_FFFF_FFFF));
    }

    #[test]
    fn xor_piece_keys_matches_naive_iteration() {
        // A few deterministic masks (edges, diagonals, random-ish constants).
        let masks: [u64; 6] = [
            0,
            0xFFFF_FFFF_FFFF_FFFFu64,
            0x0000_0000_0000_00FFu64,
            0x0102_0408_1020_4080u64,
            0x8040_2010_0804_0201u64,
            0x1234_5678_9ABC_DEF0u64,
        ];

        for &m in &masks {
            for color in [Color::Black, Color::White] {
                let mut naive = 0u64;
                let mut bb = m;
                while bb != 0 {
                    let sq = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    naive ^= piece_key(color, sq);
                }
                assert_eq!(xor_piece_keys(color, m), naive, "color={color:?} mask={m:016x}");
            }
        }
    }
}
