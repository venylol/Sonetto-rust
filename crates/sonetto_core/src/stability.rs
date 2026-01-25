//! Stability-based bounds (Stage 3)
//!
//! This module implements a **very cheap, provably safe** subset of stable discs
//! and derives **game-theoretic score bounds** from it.
//!
//! ## Why stable discs give provable bounds
//! In Othello, a disc is *stable* if it can never be flipped for the rest of the game,
//! no matter how both players move.
//!
//! If we know a (possibly incomplete) set of opponent stable discs `S_opp`:
//! - those discs are guaranteed to remain opponent discs at the end.
//! - therefore the side-to-move can occupy **at most** the other `64 - |S_opp|` squares.
//! - the final disc difference (me - opp) is thus upper bounded by:
//!   `(64 - |S_opp|) - |S_opp| = 64 - 2|S_opp|`.
//!
//! Symmetrically, if we know a (possibly incomplete) set of own stable discs `S_me`:
//! - those discs are guaranteed to remain ours at the end.
//! - therefore the opponent can occupy **at most** the other `64 - |S_me|` squares.
//! - the final disc difference (me - opp) is thus lower bounded by:
//!   `|S_me| - (64 - |S_me|) = 2|S_me| - 64`.
//!
//! ### Critical safety property
//! To be safe for alpha-beta pruning, `S_me` / `S_opp` must be a **subset** of the
//! true stable discs. Under-approximations are always safe (bounds become looser),
//! but *over-approximations* could prune incorrectly.
//!
//! ## Stable subset implemented here (P1-3)
//! We conservatively compute a subset of stable discs **on the four edges only**.
//!
//! Key observation: an **edge disc can only be flipped along that edge** (because
//! in all other directions the line immediately leaves the board, so it cannot be
//! bracketed on both sides).
//!
//! Therefore, edge stability can be computed by analysing each edge as an
//! independent 1D "Othello line" of length 8.
//!
//! We precompute a table:
//! `STAB8[me8][opp8] -> stable8` (8-bit mask)
//!
//! using a tiny recursion identical in spirit to Egaroucid's `calc_stability_line`.
//! It is intentionally conservative because it allows "moves" on any empty square
//! even if they would be illegal in real Othello — allowing extra moves can only
//! make the detected stable set **smaller**, which is still safe.
//!
//! At runtime, stability bounds cost:
//! - 4 table lookups (top/bottom/left/right)
//! - a couple of bit shuffles
//! - no heap allocation after the first initialization.

use std::sync::OnceLock;

use crate::board::Board;
use crate::score::{Score, SCALE};

/// Precomputed edge-stability helpers.
struct EdgeStabTables {
    /// Flat table: idx = (me8<<8) | opp8.
    /// Value is an 8-bit mask of squares that are stable *on that edge*.
    stab8: Box<[u8]>,
    /// Map an 8-bit mask to a file-A bitboard (bits 0,8,16,...,56).
    col_map: [u64; 256],
}

static EDGE_STAB: OnceLock<EdgeStabTables> = OnceLock::new();

#[inline(always)]
fn edge_stab_tables() -> &'static EdgeStabTables {
    EDGE_STAB.get_or_init(init_edge_stab_tables)
}

/// Warm up the stability tables (edge table + column map).
///
/// This is useful to avoid a first-search "stall" in UI builds.
#[inline]
pub fn warm_up_stability_tables() {
    let _ = edge_stab_tables();
}

// -----------------------------------------------------------------------------
// Edge stability precomputation (8-bit line game)
// -----------------------------------------------------------------------------

/// Apply a "move" on a 1D line of length 8.
///
/// This is intentionally permissive (does not require that any disc is flipped),
/// matching the conservative reference approach.
#[inline(always)]
fn probably_move_line(p: u8, o: u8, place: u8) -> (u8, u8) {
    debug_assert!(place < 8);

    let mut np: u8 = p | (1u8 << place);

    // ---- Left direction ----
    if place > 0 {
        let mut i: i32 = (place as i32) - 1;
        while i > 0 && ((o >> (i as u32)) & 1) != 0 {
            i -= 1;
        }
        if ((p >> (i as u32)) & 1) != 0 {
            let mut j: i32 = (place as i32) - 1;
            while j > i {
                np ^= 1u8 << (j as u32);
                j -= 1;
            }
        }
    }

    // ---- Right direction ----
    if place < 7 {
        let mut i: u8 = place + 1;
        while i < 7 && ((o >> (i as u32)) & 1) != 0 {
            i += 1;
        }
        if ((p >> (i as u32)) & 1) != 0 {
            let mut j: u8 = place + 1;
            while j < i {
                np ^= 1u8 << (j as u32);
                j += 1;
            }
        }
    }

    let no: u8 = o & !np;
    (np, no)
}

/// Compute stable squares on a 1D 8-square edge.
///
/// Returns an 8-bit mask of squares that are stable (never change ownership)
/// under the permissive line-game model.
#[inline(always)]
fn calc_stability_line(b: u8, w: u8, memo: &mut [[u8; 256]; 256], seen: &mut [[bool; 256]; 256]) -> u8 {
    let bi = b as usize;
    let wi = w as usize;

    if seen[bi][wi] {
        return memo[bi][wi];
    }
    seen[bi][wi] = true;

    // Invalid overlap: return empty (safe).
    if (b & w) != 0 {
        memo[bi][wi] = 0;
        return 0;
    }

    let mut res: u8 = b | w;
    let empties: u8 = !(b | w);

    for place in 0u8..8 {
        if ((empties >> place) & 1) != 0 {
            // "Black" plays.
            let (nb, nw) = probably_move_line(b, w, place);
            res &= b | nw;
            res &= calc_stability_line(nb, nw, memo, seen);

            // "White" plays.
            let (nw2, nb2) = probably_move_line(w, b, place);
            res &= w | nb2;
            res &= calc_stability_line(nb2, nw2, memo, seen);
        }
    }

    memo[bi][wi] = res;
    res
}

fn init_edge_stab_tables() -> EdgeStabTables {
    // Memoization arrays for the 8-bit recursion.
    let mut memo = [[0u8; 256]; 256];
    let mut seen = [[false; 256]; 256];

    // Ensure all states are computed.
    for b in 0u16..=255 {
        for w in 0u16..=255 {
            let _ = calc_stability_line(b as u8, w as u8, &mut memo, &mut seen);
        }
    }

    // Flatten into a cache-friendly table.
    let mut stab8 = vec![0u8; 256 * 256];
    for b in 0usize..256 {
        for w in 0usize..256 {
            stab8[(b << 8) | w] = memo[b][w];
        }
    }

    // Build u8 -> file-A bitboard map.
    let mut col_map = [0u64; 256];
    for x in 0u16..=255 {
        let mut bb = 0u64;
        let m = x as u8;
        for r in 0u8..8 {
            if ((m >> r) & 1) != 0 {
                bb |= 1u64 << ((r as u64) * 8);
            }
        }
        col_map[x as usize] = bb;
    }

    EdgeStabTables {
        stab8: stab8.into_boxed_slice(),
        col_map,
    }
}

#[inline(always)]
fn stab8_lookup(me8: u8, opp8: u8) -> u8 {
    debug_assert_eq!(me8 & opp8, 0);
    let idx = ((me8 as usize) << 8) | (opp8 as usize);
    edge_stab_tables().stab8[idx]
}

/// Extract one file (column) as an 8-bit mask (row0 in bit0, row7 in bit7).
///
/// This is a fast gather using a well-known multiplication trick.
#[inline(always)]
fn join_col8(bb: u64, col: u8) -> u8 {
    debug_assert!(col < 8);
    // Isolate the file bits into the LSB of each byte.
    let x = (bb >> (col as u32)) & 0x0101_0101_0101_0101u64;
    // Pack those 8 bits into the top byte.
    ((x.wrapping_mul(0x0102_0408_1020_4080u64) >> 56) & 0xFF) as u8
}

/// Edge stability (Sensei): stable squares on the 4 edges (union of both colors).
///
/// This is used as a seed in the full-board stable-disc propagation.
#[inline(always)]
fn stable_disks_edges(player: u64, opponent: u64) -> u64 {
    // Bottom rank A1..H1 => bits 0..7
    let p0 = (player & 0xFF) as u8;
    let o0 = (opponent & 0xFF) as u8;
    let mut stable: u64 = (stab8_lookup(p0, o0) as u64) << 0;

    // Top rank A8..H8 => bits 56..63
    let p7 = ((player >> 56) & 0xFF) as u8;
    let o7 = ((opponent >> 56) & 0xFF) as u8;
    stable |= (stab8_lookup(p7, o7) as u64) << 56;

    // File A / File H: use col_map (file A), shift for file H.
    let col_map = &edge_stab_tables().col_map;

    let p_col0 = join_col8(player, 0);
    let o_col0 = join_col8(opponent, 0);
    stable |= col_map[stab8_lookup(p_col0, o_col0) as usize];

    let p_col7 = join_col8(player, 7);
    let o_col7 = join_col8(opponent, 7);
    stable |= col_map[stab8_lookup(p_col7, o_col7) as usize] << 7;

    stable
}

// -----------------------------------------------------------------------------
// Public API: safe stable subset + bounds
// -----------------------------------------------------------------------------


// --- Full line detectors (Sensei-style bit propagation) ---

// Column/file masks for Sonetto's mapping (bit 0 == A1).
const NOT_A: u64 = 0xfefefefefefefefe;
const NOT_H: u64 = 0x7f7f7f7f7f7f7f7f;
const NOT_AB: u64 = 0xfcfcfcfcfcfcfcfc;
const NOT_GH: u64 = 0x3f3f3f3f3f3f3f3f;
const NOT_ABCD: u64 = 0xf0f0f0f0f0f0f0f0;
const NOT_EFGH: u64 = 0x0f0f0f0f0f0f0f0f;

// Board edge mask: first/last rank + first/last file.
const EDGE_MASK: u64 = 0xff818181818181ff;
const NON_EDGE_MASK: u64 = !EDGE_MASK;

#[inline(always)]
fn full_rows_mask(empty: u64) -> u64 {
    // Spread empties across their entire row (both directions), then invert.
    let mut empty_e = empty | ((empty << 1) & NOT_A);
    empty_e |= (empty_e << 2) & NOT_AB;
    empty_e |= (empty_e << 4) & NOT_ABCD;

    let mut empty_w = empty | ((empty >> 1) & NOT_H);
    empty_w |= (empty_w >> 2) & NOT_GH;
    empty_w |= (empty_w >> 4) & NOT_EFGH;

    !(empty_e | empty_w)
}

#[inline(always)]
fn full_cols_mask(empty: u64) -> u64 {
    // No wrap is possible with +/-8 shifts.
    let mut empty_n = empty | (empty << 8);
    empty_n |= empty_n << 16;
    empty_n |= empty_n << 32;

    let mut empty_s = empty | (empty >> 8);
    empty_s |= empty_s >> 16;
    empty_s |= empty_s >> 32;

    !(empty_n | empty_s)
}

#[inline(always)]
fn full_diags7_mask(empty: u64) -> u64 {
    // Diagonals with step 7 (NW<->SE). Masks prevent row-wrap across files.
    let mut empty_nw = empty | ((empty << 7) & NOT_H);
    empty_nw |= (empty_nw << 14) & NOT_GH;
    empty_nw |= (empty_nw << 28) & NOT_EFGH;

    let mut empty_se = empty | ((empty >> 7) & NOT_A);
    empty_se |= (empty_se >> 14) & NOT_AB;
    empty_se |= (empty_se >> 28) & NOT_ABCD;

    !(empty_nw | empty_se)
}

#[inline(always)]
fn full_diags9_mask(empty: u64) -> u64 {
    // Diagonals with step 9 (NE<->SW).
    let mut empty_ne = empty | ((empty << 9) & NOT_A);
    empty_ne |= (empty_ne << 18) & NOT_AB;
    empty_ne |= (empty_ne << 36) & NOT_ABCD;

    let mut empty_sw = empty | ((empty >> 9) & NOT_H);
    empty_sw |= (empty_sw >> 18) & NOT_GH;
    empty_sw |= (empty_sw >> 36) & NOT_EFGH;

    !(empty_ne | empty_sw)
}

/// Return a **definitely-stable** subset of the given side's discs.
///
/// This is a conservative (under-approx) stable-disc detector used for pruning.
/// It follows the same high-level approach as Sensei's `GetStableDisks`:
/// 1) exact stable edge discs (via the edge table),
/// 2) add squares that lie on a *full* row/column/diagonal (no empties),
/// 3) iteratively propagate stability inward along fully occupied lines.
///
/// Callers that need both sides should call this twice (swap arguments).
///
/// Implementation note:
/// Sensei's original bitboard mapping differs from Sonetto's (bit 0 == A1 here),
/// so the "full line" bit-propagation helpers below are adapted for Sonetto's
/// mapping and verified against a reference implementation.
#[inline(always)]
pub fn stable_disks_definitely(color_bits: u64, opp_bits: u64) -> u64 {
    let empty = !(color_bits | opp_bits);

    // Exact on edges (table-driven).
    let stable_edges = stable_disks_edges(color_bits, opp_bits);

    // "Full line" masks (no empties anywhere on the line).
    let full_rows = full_rows_mask(empty);
    let full_cols = full_cols_mask(empty);
    let full_diag7 = full_diags7_mask(empty);
    let full_diag9 = full_diags9_mask(empty);

    // Sensei: squares that are simultaneously on a full row, column, and both diagonals
    // are immediately stable.
    let stable_full = full_rows & full_cols & full_diag7 & full_diag9;

    // Start from edge-stable discs + full-line intersections, restricted to our color.
    let mut stable = (stable_edges | stable_full) & color_bits;

    // Propagate stability inward. Edges are already handled by the edge table.
    // Restricting to non-edge also avoids wrap-around artifacts from bit shifts.
    let mut newly_added = stable;
    while newly_added != 0 {
        newly_added = ((stable << 1) | (stable >> 1) | full_rows)
            & ((stable << 8) | (stable >> 8) | full_cols)
            & ((stable << 7) | (stable >> 7) | full_diag7)
            & ((stable << 9) | (stable >> 9) | full_diag9);

        newly_added &= color_bits & NON_EDGE_MASK & !stable;
        stable |= newly_added;
    }

    stable
}


/// Compute a provable `(lower, upper)` bound for the **side-to-move**.
///
/// Bounds are returned in the engine's `Score` units (scaled by [`SCALE`]).
///
/// - `upper = (64 - 2*stable_opp) * SCALE`
/// - `lower = (2*stable_me - 64) * SCALE`
#[inline(always)]
pub fn stability_bounds_for_side_to_move(board: &Board) -> (Score, Score) {
    let me = board.player;
    let opp = board.opponent;
    stability_bounds_for_bitboards(me, opp)
}

/// Same as [`stability_bounds_for_side_to_move`], but with explicit `(me, opp)`.
#[inline(always)]
pub fn stability_bounds_for_bitboards(me: u64, opp: u64) -> (Score, Score) {
    let stable_me = stable_disks_definitely(me, opp);
    let stable_opp = stable_disks_definitely(opp, me);

    let me_cnt: i32 = stable_me.count_ones() as i32;
    let opp_cnt: i32 = stable_opp.count_ones() as i32;

    // Disc-difference bounds (unscaled).
    let upper_disc: i32 = 64 - 2 * opp_cnt;
    let lower_disc: i32 = 2 * me_cnt - 64;

    (lower_disc * SCALE, upper_disc * SCALE)
}

/// Convenience: only the stability-based **upper** bound for the side-to-move.
#[inline(always)]
pub fn stability_upper_bound_for_side_to_move(board: &Board) -> Score {
    let (_lo, hi) = stability_bounds_for_side_to_move(board);
    hi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Color};

    #[test]
    fn stable_disks_is_subset_of_color_bits() {
        // Random-ish pattern: only check subset property.
        let color = 0x0123_4567_89AB_CDEFu64;
        let opp = 0x0FED_CBA9_8765_4321u64 & !color;
        let s = stable_disks_definitely(color, opp);
        assert_eq!(s & !color, 0, "stable subset must never include opponent/empty bits");
    }

    #[test]
    fn start_position_has_full_range_bounds() {
        let b = Board::new_start(0);
        let (lo, hi) = stability_bounds_for_side_to_move(&b);
        assert_eq!(lo, -64 * SCALE);
        assert_eq!(hi, 64 * SCALE);
    }

    #[test]
    fn corner_chain_on_top_edge_is_detected_stable() {
        // Construct a position where Black owns A1,B1,C1.
        // We do not care about legality here; stable detection is purely geometric.
        let black = (1u64 << 0) | (1u64 << 1) | (1u64 << 2);
        let white = 0u64;

        let mut b = Board::new_start(0);
        b.side = Color::Black;
        b.player = black;
        b.opponent = white;
        b.empty_count = 64 - (black | white).count_ones() as u8;

        let s = stable_disks_definitely(black, white);
        assert_eq!(s & black, black, "A1-connected top-edge chain should be stable");

        let (_lo, hi) = stability_bounds_for_side_to_move(&b);
        // Opponent has 0 stable discs => upper bound is still +64.
        assert_eq!(hi, 64 * SCALE);
    }

    #[test]
    fn join_col8_roundtrip_matches_manual_gather() {
        // Spot-check the gather logic against a slow reference.
        for col in 0u8..8 {
            let bb = 0x0123_4567_89AB_CDEFu64;
            let fast = join_col8(bb, col);
            let mut slow = 0u8;
            for r in 0u8..8 {
                let bit = (bb >> (col as u32 + 8 * (r as u32))) & 1;
                slow |= (bit as u8) << r;
            }
            assert_eq!(fast, slow);
        }
    }
}
