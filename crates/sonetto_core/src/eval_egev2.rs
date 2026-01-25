//! Egaroucid-compatible evaluation ("egev2")
//!
//! This module aligns Sonetto's evaluation semantics with:
//! - `Egaroucid/src/engine/evaluate_generic.hpp`
//! - `Egaroucid/src/tools/evaluation/evaluation_definition_20241125_1_7_5.hpp`
//!
//! Key properties:
//! - 60 phases: `phase = clamp((discs - 4), 0..59)`
//! - 16 patterns, each with 4 symmetry features => 64 feature lookups
//! - plus `eval_num_arr[phase][num_player_discs]`
//! - raw sum is in "STEP" units; final score is `round_nearest(raw / 32)` with
//!   half-away-from-zero behavior (matches C++ `mid_evaluate`).

use crate::board::{Board, Color};
use crate::features::occ::{OccMap, Occurrence};
use crate::features::swap::{pow3_u16, pow3_u32, SwapTables};
use std::sync::{Arc, OnceLock};

pub const N_PHASES: usize = 60;
pub const WEIGHT_PHASES: usize = N_PHASES;

pub const STEP: i32 = 32;
pub const STEP_2: i32 = 16;
pub const SCORE_MAX: i32 = 64;

pub const N_PATTERNS: usize = 16;
pub const N_PATTERN_FEATURES: usize = 64;
pub const MAX_PATTERN_CELLS: usize = 10;

pub const EVAL_NUM_LEN: usize = 65; // num_player_discs: 0..64

/// Pattern sizes (cells per pattern), from Egaroucid.
pub const PATTERN_SIZES: [usize; N_PATTERNS] = [
    8, 9, 8, 9, 8, 9, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10,
];

/// Reversal digit mapping for each base pattern.
///
/// Direct port of Egaroucid's `adj_rev_patterns` from:
/// `src/tools/evaluation/evaluation_definition_20241125_1_7_5.hpp`.
///
/// For a pattern of length `n = PATTERN_SIZES[p]`, the reversal permutation is
/// `REV_PATTERNS[p][0..n]`.
pub const REV_PATTERNS: [[u8; MAX_PATTERN_CELLS]; N_PATTERNS] = [
    // 0: hv2 (8)
    [7, 6, 5, 4, 3, 2, 1, 0, 0, 0],
    // 1: hv3 (9)
    [8, 7, 6, 5, 4, 3, 2, 1, 0, 0],
    // 2: hv4 (8)
    [7, 6, 5, 4, 3, 2, 1, 0, 0, 0],
    // 3: diag4 (9)
    [8, 7, 6, 5, 4, 3, 2, 1, 0, 0],
    // 4: edge2X (8)
    [7, 6, 5, 4, 3, 2, 1, 0, 0, 0],
    // 5: corner9 (9)
    [0, 3, 6, 1, 4, 7, 2, 5, 8, 0],
    // 6: triangle7 (7)
    [6, 5, 4, 3, 2, 1, 0, 0, 0, 0],
    // 7: edge_2Y (10)
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
    // 8: edge_2Z (10)
    [7, 6, 5, 4, 3, 2, 1, 0, 9, 8],
    // 9: corner_2x5 (10)
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    // 10: diag5 (10)
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    // 11: diag6 (10)
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    // 12: diag7 (10)
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    // 13: diag8 (10)
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    // 14: corner_3x3 (10)
    [2, 1, 0, 5, 4, 3, 8, 7, 6, 9],
    // 15: edge_3x3 (10)
    [8, 7, 6, 5, 4, 3, 2, 1, 0, 9],
];

/// 3^PATTERN_SIZES (table length) for each pattern.
pub const PATTERN_POW3: [usize; N_PATTERNS] = [
    6561, 19683, 6561, 19683, 6561, 19683, 2187, 59049, 59049, 59049, 59049, 59049, 59049, 59049,
    59049, 59049,
];

/// Prefix sums of PATTERN_POW3 within a phase.
pub const PATTERN_OFFSETS: [usize; N_PATTERNS] = [
    0,       // p0
    6561,    // p1
    26244,   // p2
    32805,   // p3
    52488,   // p4
    59049,   // p5
    78732,   // p6
    80919,   // p7
    139968,  // p8
    199017,  // p9
    258066,  // p10
    317115,  // p11
    376164,  // p12
    435213,  // p13
    494262,  // p14
    553311,  // p15
];

pub const PATTERN_PARAMS_PER_PHASE: usize = 612_360;
pub const PARAMS_PER_PHASE: usize = PATTERN_PARAMS_PER_PHASE + EVAL_NUM_LEN;
pub const EXPECTED_PARAMS_LEN: usize = PARAMS_PER_PHASE * N_PHASES;

/// Maps the 64 symmetry features to their "base pattern" index (0..15).
pub const FEATURE_TO_PATTERN: [usize; N_PATTERN_FEATURES] = [
    0, 0, 0, 0, // 0
    1, 1, 1, 1, // 1
    2, 2, 2, 2, // 2
    3, 3, 3, 3, // 3
    4, 4, 4, 4, // 4
    5, 5, 5, 5, // 5
    6, 6, 6, 6, // 6
    7, 7, 7, 7, // 7
    8, 8, 8, 8, // 8
    9, 9, 9, 9, // 9
    10, 10, 10, 10, // 10
    11, 11, 11, 11, // 11
    12, 12, 12, 12, // 12
    13, 13, 13, 13, // 13
    14, 14, 14, 14, // 14
    15, 15, 15, 15, // 15
];

#[derive(Clone, Copy, Debug)]
pub struct FeatureToCoord {
    pub n_cells: u8,
    pub cells: [u8; MAX_PATTERN_CELLS],
}

/// A compact representation of the **active** (phase + feature indices) inputs
/// used by the EGEV2 evaluator.
///
/// This exists primarily to enable **incremental training updates** without
/// re-deriving feature semantics in downstream crates.
#[derive(Clone, Copy, Debug)]
pub struct ActiveFeatures {
    /// Phase index (0..59).
    pub phase: usize,
    /// Number of stones of the side-to-move.
    pub num_player_discs: u8,
    /// Pattern table indices for the 64 symmetry features.
    pub idx_by_feature: [u16; N_PATTERN_FEATURES],
}

// Generated from:
//   Egaroucid/src/tools/evaluation/evaluation_definition_20241125_1_7_5.hpp
// Each `cells` entry is a Sonetto bitpos (row-major, A1=0 .. H8=63).
pub const EGEV2_FEATURE_TO_COORD: [FeatureToCoord; N_PATTERN_FEATURES] = [
    FeatureToCoord { n_cells: 8u8, cells: [8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8, 15u8, 0u8, 0u8] }, // 0
    FeatureToCoord { n_cells: 8u8, cells: [1u8, 9u8, 17u8, 25u8, 33u8, 41u8, 49u8, 57u8, 0u8, 0u8] }, // 1
    FeatureToCoord { n_cells: 8u8, cells: [48u8, 49u8, 50u8, 51u8, 52u8, 53u8, 54u8, 55u8, 0u8, 0u8] }, // 2
    FeatureToCoord { n_cells: 8u8, cells: [6u8, 14u8, 22u8, 30u8, 38u8, 46u8, 54u8, 62u8, 0u8, 0u8] }, // 3
    FeatureToCoord { n_cells: 9u8, cells: [1u8, 2u8, 11u8, 20u8, 14u8, 29u8, 38u8, 47u8, 55u8, 0u8] }, // 4
    FeatureToCoord { n_cells: 9u8, cells: [15u8, 23u8, 30u8, 37u8, 54u8, 44u8, 51u8, 58u8, 57u8, 0u8] }, // 5
    FeatureToCoord { n_cells: 9u8, cells: [62u8, 61u8, 52u8, 43u8, 49u8, 34u8, 25u8, 16u8, 8u8, 0u8] }, // 6
    FeatureToCoord { n_cells: 9u8, cells: [48u8, 40u8, 33u8, 26u8, 9u8, 19u8, 12u8, 5u8, 6u8, 0u8] }, // 7
    FeatureToCoord { n_cells: 8u8, cells: [16u8, 17u8, 18u8, 19u8, 20u8, 21u8, 22u8, 23u8, 0u8, 0u8] }, // 8
    FeatureToCoord { n_cells: 8u8, cells: [2u8, 10u8, 18u8, 26u8, 34u8, 42u8, 50u8, 58u8, 0u8, 0u8] }, // 9
    FeatureToCoord { n_cells: 8u8, cells: [40u8, 41u8, 42u8, 43u8, 44u8, 45u8, 46u8, 47u8, 0u8, 0u8] }, // 10
    FeatureToCoord { n_cells: 8u8, cells: [5u8, 13u8, 21u8, 29u8, 37u8, 45u8, 53u8, 61u8, 0u8, 0u8] }, // 11
    FeatureToCoord { n_cells: 9u8, cells: [0u8, 1u8, 10u8, 19u8, 28u8, 37u8, 46u8, 55u8, 63u8, 0u8] }, // 12
    FeatureToCoord { n_cells: 9u8, cells: [7u8, 15u8, 22u8, 29u8, 36u8, 43u8, 50u8, 57u8, 56u8, 0u8] }, // 13
    FeatureToCoord { n_cells: 9u8, cells: [63u8, 62u8, 53u8, 44u8, 35u8, 26u8, 17u8, 8u8, 0u8, 0u8] }, // 14
    FeatureToCoord { n_cells: 9u8, cells: [56u8, 48u8, 41u8, 34u8, 27u8, 20u8, 13u8, 6u8, 7u8, 0u8] }, // 15
    FeatureToCoord { n_cells: 8u8, cells: [24u8, 25u8, 26u8, 27u8, 28u8, 29u8, 30u8, 31u8, 0u8, 0u8] }, // 16
    FeatureToCoord { n_cells: 8u8, cells: [3u8, 11u8, 19u8, 27u8, 35u8, 43u8, 51u8, 59u8, 0u8, 0u8] }, // 17
    FeatureToCoord { n_cells: 8u8, cells: [32u8, 33u8, 34u8, 35u8, 36u8, 37u8, 38u8, 39u8, 0u8, 0u8] }, // 18
    FeatureToCoord { n_cells: 8u8, cells: [4u8, 12u8, 20u8, 28u8, 36u8, 44u8, 52u8, 60u8, 0u8, 0u8] }, // 19
    FeatureToCoord { n_cells: 9u8, cells: [0u8, 1u8, 2u8, 8u8, 9u8, 10u8, 16u8, 17u8, 18u8, 0u8] }, // 20
    FeatureToCoord { n_cells: 9u8, cells: [7u8, 6u8, 5u8, 15u8, 14u8, 13u8, 23u8, 22u8, 21u8, 0u8] }, // 21
    FeatureToCoord { n_cells: 9u8, cells: [56u8, 57u8, 58u8, 48u8, 49u8, 50u8, 40u8, 41u8, 42u8, 0u8] }, // 22
    FeatureToCoord { n_cells: 9u8, cells: [63u8, 62u8, 61u8, 55u8, 54u8, 53u8, 47u8, 46u8, 45u8, 0u8] }, // 23
    FeatureToCoord { n_cells: 7u8, cells: [9u8, 3u8, 12u8, 21u8, 30u8, 39u8, 54u8, 0u8, 0u8, 0u8] }, // 24
    FeatureToCoord { n_cells: 7u8, cells: [14u8, 31u8, 38u8, 45u8, 52u8, 59u8, 49u8, 0u8, 0u8, 0u8] }, // 25
    FeatureToCoord { n_cells: 7u8, cells: [54u8, 60u8, 51u8, 42u8, 33u8, 24u8, 9u8, 0u8, 0u8, 0u8] }, // 26
    FeatureToCoord { n_cells: 7u8, cells: [49u8, 32u8, 25u8, 18u8, 11u8, 4u8, 14u8, 0u8, 0u8, 0u8] }, // 27
    FeatureToCoord { n_cells: 10u8, cells: [0u8, 9u8, 18u8, 27u8, 36u8, 45u8, 54u8, 63u8, 8u8, 1u8] }, // 28
    FeatureToCoord { n_cells: 10u8, cells: [63u8, 54u8, 45u8, 36u8, 27u8, 18u8, 9u8, 0u8, 55u8, 62u8] }, // 29
    FeatureToCoord { n_cells: 10u8, cells: [7u8, 14u8, 21u8, 28u8, 35u8, 42u8, 49u8, 56u8, 15u8, 6u8] }, // 30
    FeatureToCoord { n_cells: 10u8, cells: [56u8, 49u8, 42u8, 35u8, 28u8, 21u8, 14u8, 7u8, 48u8, 57u8] }, // 31
    FeatureToCoord { n_cells: 10u8, cells: [9u8, 0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 14u8] }, // 32
    FeatureToCoord { n_cells: 10u8, cells: [9u8, 0u8, 8u8, 16u8, 24u8, 32u8, 40u8, 48u8, 56u8, 49u8] }, // 33
    FeatureToCoord { n_cells: 10u8, cells: [49u8, 56u8, 57u8, 58u8, 59u8, 60u8, 61u8, 62u8, 63u8, 54u8] }, // 34
    FeatureToCoord { n_cells: 10u8, cells: [14u8, 7u8, 15u8, 23u8, 31u8, 39u8, 47u8, 55u8, 63u8, 54u8] }, // 35
    FeatureToCoord { n_cells: 10u8, cells: [0u8, 1u8, 2u8, 3u8, 8u8, 9u8, 10u8, 16u8, 17u8, 24u8] }, // 36
    FeatureToCoord { n_cells: 10u8, cells: [7u8, 6u8, 5u8, 4u8, 15u8, 14u8, 13u8, 23u8, 22u8, 31u8] }, // 37
    FeatureToCoord { n_cells: 10u8, cells: [56u8, 57u8, 58u8, 59u8, 48u8, 49u8, 50u8, 40u8, 41u8, 32u8] }, // 38
    FeatureToCoord { n_cells: 10u8, cells: [63u8, 62u8, 61u8, 60u8, 55u8, 54u8, 53u8, 47u8, 46u8, 39u8] }, // 39
    FeatureToCoord { n_cells: 10u8, cells: [0u8, 2u8, 3u8, 4u8, 5u8, 7u8, 10u8, 11u8, 12u8, 13u8] }, // 40
    FeatureToCoord { n_cells: 10u8, cells: [0u8, 16u8, 24u8, 32u8, 40u8, 56u8, 17u8, 25u8, 33u8, 41u8] }, // 41
    FeatureToCoord { n_cells: 10u8, cells: [56u8, 58u8, 59u8, 60u8, 61u8, 63u8, 50u8, 51u8, 52u8, 53u8] }, // 42
    FeatureToCoord { n_cells: 10u8, cells: [7u8, 23u8, 31u8, 39u8, 47u8, 63u8, 22u8, 30u8, 38u8, 46u8] }, // 43
    FeatureToCoord { n_cells: 10u8, cells: [0u8, 9u8, 18u8, 27u8, 1u8, 10u8, 19u8, 8u8, 17u8, 26u8] }, // 44
    FeatureToCoord { n_cells: 10u8, cells: [7u8, 14u8, 21u8, 28u8, 6u8, 13u8, 20u8, 15u8, 22u8, 29u8] }, // 45
    FeatureToCoord { n_cells: 10u8, cells: [56u8, 49u8, 42u8, 35u8, 57u8, 50u8, 43u8, 48u8, 41u8, 34u8] }, // 46
    FeatureToCoord { n_cells: 10u8, cells: [63u8, 54u8, 45u8, 36u8, 62u8, 53u8, 44u8, 55u8, 46u8, 37u8] }, // 47
    FeatureToCoord { n_cells: 10u8, cells: [10u8, 0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 13u8] }, // 48
    FeatureToCoord { n_cells: 10u8, cells: [17u8, 0u8, 8u8, 16u8, 24u8, 32u8, 40u8, 48u8, 56u8, 41u8] }, // 49
    FeatureToCoord { n_cells: 10u8, cells: [50u8, 56u8, 57u8, 58u8, 59u8, 60u8, 61u8, 62u8, 63u8, 53u8] }, // 50
    FeatureToCoord { n_cells: 10u8, cells: [22u8, 7u8, 15u8, 23u8, 31u8, 39u8, 47u8, 55u8, 63u8, 46u8] }, // 51
    FeatureToCoord { n_cells: 10u8, cells: [0u8, 1u8, 2u8, 3u8, 4u8, 8u8, 9u8, 16u8, 24u8, 32u8] }, // 52
    FeatureToCoord { n_cells: 10u8, cells: [7u8, 6u8, 5u8, 4u8, 3u8, 15u8, 14u8, 23u8, 31u8, 39u8] }, // 53
    FeatureToCoord { n_cells: 10u8, cells: [56u8, 57u8, 58u8, 59u8, 60u8, 48u8, 49u8, 40u8, 32u8, 24u8] }, // 54
    FeatureToCoord { n_cells: 10u8, cells: [63u8, 62u8, 61u8, 60u8, 59u8, 55u8, 54u8, 47u8, 39u8, 31u8] }, // 55
    FeatureToCoord { n_cells: 10u8, cells: [0u8, 1u8, 8u8, 9u8, 10u8, 11u8, 17u8, 18u8, 25u8, 27u8] }, // 56
    FeatureToCoord { n_cells: 10u8, cells: [7u8, 6u8, 15u8, 14u8, 13u8, 12u8, 22u8, 21u8, 30u8, 28u8] }, // 57
    FeatureToCoord { n_cells: 10u8, cells: [56u8, 57u8, 48u8, 49u8, 50u8, 51u8, 41u8, 42u8, 33u8, 35u8] }, // 58
    FeatureToCoord { n_cells: 10u8, cells: [63u8, 62u8, 55u8, 54u8, 53u8, 52u8, 46u8, 45u8, 38u8, 36u8] }, // 59
    FeatureToCoord { n_cells: 10u8, cells: [42u8, 43u8, 51u8, 59u8, 58u8, 61u8, 60u8, 52u8, 44u8, 45u8] }, // 60
    FeatureToCoord { n_cells: 10u8, cells: [18u8, 26u8, 25u8, 24u8, 16u8, 40u8, 32u8, 33u8, 34u8, 42u8] }, // 61
    FeatureToCoord { n_cells: 10u8, cells: [21u8, 20u8, 12u8, 4u8, 5u8, 2u8, 3u8, 11u8, 19u8, 18u8] }, // 62
    FeatureToCoord { n_cells: 10u8, cells: [45u8, 37u8, 38u8, 39u8, 47u8, 23u8, 31u8, 30u8, 29u8, 21u8] }, // 63
];

#[derive(Clone, Debug, Default)]
pub struct FeatureDefs {
    _priv: (),
}

// ---------------------------------------------------------------------------
// P1-2: Static OccMap for EGEV2
// ---------------------------------------------------------------------------
//
// Egaroucid keeps a precomputed mapping `coord_to_feature[sq] -> (feature_idx, x=3^pos)`.
// Sonetto's equivalent is `OccMap`.
//
// For the EGEV2 64 symmetry features, we can generate the OccMap entirely at
// compile-time (no Vec allocations at engine init).

const fn egev2_total_occurrences() -> usize {
    let mut sum = 0usize;
    let mut fi = 0usize;
    while fi < N_PATTERN_FEATURES {
        sum += EGEV2_FEATURE_TO_COORD[fi].n_cells as usize;
        fi += 1;
    }
    sum
}

/// Total number of (square,feature) occurrences in the 64 EGEV2 features.
pub const EGEV2_TOTAL_OCC: usize = egev2_total_occurrences();

const fn build_egev2_occ_data() -> ([u16; 65], [Occurrence; EGEV2_TOTAL_OCC]) {
    let mut off = [0u16; 65];
    let mut flat = [Occurrence { feature_idx: 0u16, pow3: 0u16 }; EGEV2_TOTAL_OCC];

    let mut idx = 0usize;
    let mut sq = 0usize;
    while sq < 64 {
        // Start offset for this square.
        off[sq] = idx as u16;

        let mut fi = 0usize;
        while fi < N_PATTERN_FEATURES {
            let f = EGEV2_FEATURE_TO_COORD[fi];
            let n = f.n_cells as usize;

            let mut j = 0usize;
            while j < n {
                if f.cells[j] as usize == sq {
                    // OccMap expects LSB-first order; Egaroucid's feature definition
                    // lists cells MSB-first, so we reverse the position index.
                    let pos = (n - 1 - j) as u8;
                    flat[idx] = Occurrence {
                        feature_idx: fi as u16,
                        pow3: pow3_u16(pos),
                    };
                    idx += 1;
                }
                j += 1;
            }

            fi += 1;
        }

        // End offset (exclusive) for this square.
        off[sq + 1] = idx as u16;
        sq += 1;
    }

    (off, flat)
}

const EGEV2_OCC_DATA: ([u16; 65], [Occurrence; EGEV2_TOTAL_OCC]) = build_egev2_occ_data();

/// `occ_off[sq]..occ_off[sq+1]` ranges into [`EGEV2_OCC_FLAT`].
pub const EGEV2_OCC_OFF: [u16; 65] = EGEV2_OCC_DATA.0;

/// Flat array of all (feature_idx, pow3) occurrences, grouped by square.
pub const EGEV2_OCC_FLAT: [Occurrence; EGEV2_TOTAL_OCC] = EGEV2_OCC_DATA.1;

/// Fully static OccMap for EGEV2 (no allocation).
pub const EGEV2_OCC: OccMap = OccMap::from_static(&EGEV2_OCC_FLAT, EGEV2_OCC_OFF);


/// Build the EGEV2 feature definition handle and the corresponding `OccMap`.
///
/// Sonetto maintains **absolute ternary pattern IDs** (`feat_id_abs`) incrementally during search.
/// For EGEV2, each feature is one of the 64 symmetry-expanded patterns defined in
/// [`EGEV2_FEATURE_TO_COORD`].
///
/// Important: `OccMap` uses base-3 powers in *least-significant-first* order.
/// Egaroucid's indexing is *most-significant-first*, so we reverse the **position index**
/// when generating the map (see `build_egev2_occ_data`).
///
/// This returns a fully static `OccMap` (no allocations).
pub fn build_sonetto_feature_defs_and_occ() -> (FeatureDefs, OccMap) {
    (FeatureDefs { _priv: () }, EGEV2_OCC)
}

#[derive(Clone, Debug, Default)]
pub struct Weights {
    /// Unzipped parameters, laid out exactly as Egaroucid expects:
    /// for each phase:
    ///   - pattern[0] table (3^size)
    ///   - ...
    ///   - pattern[15] table (3^size)
    ///   - eval_num[0..64]
    /// NOTE: this is intentionally shared (`Arc`) so that cloning a `Searcher`
    /// or spawning parallel workers does **not** duplicate the ~70MiB weight
    /// buffer.
    ///
    /// This eliminates major allocator pressure / GC-like stalls in WASM and
    /// thread-heavy native analysis.
    pub params: Arc<Vec<i16>>,
}

impl Weights {
    /// Create an empty weights object. All missing parameters read as 0.
    pub fn zeroed() -> Self {
        Self {
            params: Arc::new(Vec::new()),
        }
    }

    /// Wrap an owned parameter vector without copying.
    #[inline]
    pub fn from_vec(v: Vec<i16>) -> Self {
        Self { params: Arc::new(v) }
    }

    /// Immutable view of the parameter buffer.
    #[inline(always)]
    pub fn as_slice(&self) -> &[i16] {
        self.params.as_slice()
    }

    /// Mutable access for trainer-style updates.
    ///
    /// If the buffer is uniquely owned, this is zero-copy.
    /// If it is shared, Arc performs a copy-on-write clone.
    #[inline]
    pub fn make_mut(&mut self) -> &mut Vec<i16> {
        Arc::make_mut(&mut self.params)
    }

    pub fn expected_len() -> usize {
        EXPECTED_PARAMS_LEN
    }

    pub fn is_valid(&self) -> bool {
        self.params.len() == EXPECTED_PARAMS_LEN
    }

    #[inline(always)]
    fn get_param(&self, idx: usize) -> i16 {
        // Safe: treat missing params as 0.
        *self.as_slice().get(idx).unwrap_or(&0)
    }

    #[inline(always)]
    pub fn get_pattern_weight(&self, phase: usize, pattern: usize, idx: usize) -> i16 {
        if phase >= N_PHASES || pattern >= N_PATTERNS {
            return 0;
        }
        let max = PATTERN_POW3[pattern];
        if idx >= max {
            return 0;
        }
        let base = phase * PARAMS_PER_PHASE + PATTERN_OFFSETS[pattern];
        self.get_param(base + idx)
    }

    #[inline(always)]
    pub fn get_eval_num_weight(&self, phase: usize, num_player_discs: usize) -> i16 {
        if phase >= N_PHASES || num_player_discs >= EVAL_NUM_LEN {
            return 0;
        }
        let base = phase * PARAMS_PER_PHASE + PATTERN_PARAMS_PER_PHASE;
        self.get_param(base + num_player_discs)
    }
}

/// Egaroucid phase mapping.
///
/// `discs` is total discs on board (4..64).
#[inline(always)]
pub fn phase_from_discs(discs: i32) -> usize {
    let mut p = discs - 4;
    if p < 0 {
        p = 0;
    } else if p > (N_PHASES as i32 - 1) {
        p = N_PHASES as i32 - 1;
    }
    p as usize
}

/// Egaroucid phase mapping from empty_count.
#[inline(always)]
pub fn phase(empty_count: u8) -> usize {
    let discs = 64i32 - empty_count as i32;
    phase_from_discs(discs)
}

#[inline(always)]
fn digit_player_opp_empty(player: u64, opp: u64, bit: u64) -> usize {
    if (player & bit) != 0 {
        0
    } else if (opp & bit) != 0 {
        1
    } else {
        2
    }
}

#[inline(always)]
fn feature_idx_for_board(player: u64, opp: u64, feat: &FeatureToCoord) -> usize {
    let mut idx: usize = 0;
    let n = feat.n_cells as usize;
    let mut j = 0;
    while j < n {
        idx *= 3;
        let sq = feat.cells[j] as u32;
        let bit = 1u64 << sq;
        idx += digit_player_opp_empty(player, opp, bit);
        j += 1;
    }
    idx
}


// ---------------------------------------------------------------------------
// Phase0 P0-3: incremental pattern indices (abs_id -> EGEV2 id)
// ---------------------------------------------------------------------------

struct AbsToEgev2Tables {
    // map[side_idx][len][abs_id] -> egev2_id
    map: [[Vec<u16>; MAX_PATTERN_CELLS + 1]; 2],
}

static ABS_TO_EGEV2: OnceLock<AbsToEgev2Tables> = OnceLock::new();

#[inline(always)]
fn abs_to_egev2_tables() -> &'static AbsToEgev2Tables {
    ABS_TO_EGEV2.get_or_init(|| {
        let map: [[Vec<u16>; MAX_PATTERN_CELLS + 1]; 2] = std::array::from_fn(|side_idx| {
            std::array::from_fn(|len| {
                if len == 0 {
                    return Vec::new();
                }
                let size = pow3_u32(len as u8) as usize;
                let mut v: Vec<u16> = vec![0u16; size];
                for abs_id in 0..size {
                    v[abs_id] = abs_to_egev2_on_the_fly(
                        if side_idx == 0 { Color::Black } else { Color::White },
                        len,
                        abs_id as u16,
                    );
                }
                v
            })
        });
        AbsToEgev2Tables { map }
    })
}

#[inline(always)]
fn abs_digit_to_egev2_digit(side: Color, abs_digit: u16) -> u16 {
    match abs_digit {
        0 => 2, // empty
        1 => if side == Color::Black { 0 } else { 1 }, // black
        2 => if side == Color::Black { 1 } else { 0 }, // white
        _ => 2,
    }
}

#[inline(always)]
fn abs_to_egev2_on_the_fly(side: Color, len: usize, abs_id: u16) -> u16 {
    // abs_id is a base-3 number in absolute digits (0=empty,1=black,2=white).
    // Convert it into Egaroucid's player-relative ternary digits.
    let mut x: u32 = abs_id as u32;
    let mut out: u32 = 0;
    let mut pow: u32 = 1;
    let mut i = 0usize;
    while i < len {
        let d = (x % 3) as u16;
        x /= 3;
        out += (abs_digit_to_egev2_digit(side, d) as u32) * pow;
        pow *= 3;
        i += 1;
    }
    out as u16
}

#[inline(always)]
pub(crate) fn abs_to_egev2_idx(side: Color, len: usize, abs_id: u16) -> usize {
    if len == 0 || len > MAX_PATTERN_CELLS {
        return 0;
    }
    let tbl = abs_to_egev2_tables();
    let v = &tbl.map[side.idx()][len];
    let a = abs_id as usize;
    if a < v.len() {
        v[a] as usize
    } else {
        // Defensive fallback: avoid panics if a malformed/buggy abs_id slips through.
        abs_to_egev2_on_the_fly(side, len, abs_id) as usize
    }
}

/// Calculate all EGEV2 feature indices for a board position.
///
/// This is a *semantic* helper: downstream crates (trainer, analysis tools)
/// can reuse Sonetto's exact feature definition and avoid re-implementing
/// ternary indexing or coordinate lists.
#[inline(always)]
pub fn calc_active_features(board: &Board) -> ActiveFeatures {
    let ph = phase(board.empty_count);

    // Keep the absolute side available for future feature variants (and for
    // parity with other helpers), but it's not currently needed here.
    let _side = board.side;
    let player = board.player;
    let opp = board.opponent;

    let mut idx_by_feature: [u16; N_PATTERN_FEATURES] = [0u16; N_PATTERN_FEATURES];
    let mut fi = 0;
    while fi < N_PATTERN_FEATURES {
        let f = &EGEV2_FEATURE_TO_COORD[fi];
        let idx = feature_idx_for_board(player, opp, f);
        debug_assert!(idx < 65536);
        idx_by_feature[fi] = idx as u16;
        fi += 1;
    }

    ActiveFeatures {
        phase: ph,
        num_player_discs: player.count_ones() as u8,
        idx_by_feature,
    }
}

/// Convert a raw STEP=32 score into a disc score using Egaroucid's semantics:
///
/// `raw -> half-away-from-zero rounding (+16/-16) -> /32 -> clamp [-64, 64]`.
#[inline(always)]
pub fn round_clamp_raw_to_disc(raw: i32) -> i32 {
    // C++: res += res >= 0 ? STEP_2 : -STEP_2; res /= STEP;
    let mut res = raw;
    res += if res >= 0 { STEP_2 } else { -STEP_2 };
    res /= STEP;

    if res > SCORE_MAX {
        SCORE_MAX
    } else if res < -SCORE_MAX {
        -SCORE_MAX
    } else {
        res
    }
}

/// Egaroucid-compatible reversed index mapping (`adj_calc_rev_idx`).
///
/// `pattern` must be a base pattern index (0..15). For out-of-range values,
/// this returns `idx` unchanged.
#[inline]
pub fn calc_rev_idx(pattern: usize, idx: u16) -> u16 {
    if pattern >= N_PATTERNS {
        return idx;
    }
    let n = PATTERN_SIZES[pattern];
    debug_assert!(n <= MAX_PATTERN_CELLS);

    // Defensive bound check.
    //
    // For valid feature indices (computed from a ternary board encoding), `idx` is always in
    // `[0, 3^n)`. However, keeping this guard makes the helper resilient when called from
    // external tooling / fuzzing / malformed datasets.
    let pow3 = PATTERN_POW3[pattern];
    if (idx as usize) >= pow3 {
        return idx;
    }

    // Decode base-3 digits (most significant first).
    let mut digits: [u8; MAX_PATTERN_CELLS] = [0u8; MAX_PATTERN_CELLS];
    let mut x = idx as usize;
    let mut pos = n;
    while pos > 0 {
        pos -= 1;
        digits[pos] = (x % 3) as u8;
        x /= 3;
    }

    // Apply the permutation.
    let mut res: usize = 0;
    let mut i = 0;
    while i < n {
        res = res * 3 + digits[REV_PATTERNS[pattern][i] as usize] as usize;
        i += 1;
    }

    res as u16
}

/// Mid-game evaluation in "disc score" units (clamped to [-64, 64]).
///
/// This matches Egaroucid's `mid_evaluate`:
/// - sum int16 weights (patterns + eval_num)
/// - round-to-nearest: `(raw + sign*16) / 32` with trunc-toward-zero division
/// - clamp to [-64, 64]
#[inline(always)]
pub fn score_disc(board: &Board, weights: &Weights) -> i32 {
    #[cfg(all(feature = "wasm_simd", target_arch = "wasm32", target_feature = "simd128"))]
    {
        // Wasm SIMD path (simd128): keep semantics identical to the scalar version,
        // but accumulate pattern weights with vector ops.
        return score_disc_wasm_simd(board, weights);
    }

    #[cfg(not(all(feature = "wasm_simd", target_arch = "wasm32", target_feature = "simd128")))]
    {
        let ph = phase(board.empty_count);

    let side = board.side;
    let player = board.player;
    let opp = board.opponent;

    // Fast path: incremental absolute pattern IDs are available.
    //
    // We track this explicitly on the board to avoid scanning the whole vector
    // on every evaluation call.
    let use_fast = board.feat_is_pattern_ids && board.feat_id_abs.len() == N_PATTERN_FEATURES;

    // PERF (high ROI): when the weight buffer is valid, index directly into the
    // flat parameter slice with unchecked bounds. This removes several layers
    // of per-feature bounds checks (`get_pattern_weight` / `get_param`) and
    // measurably improves NPS in the hot leaf-eval path.
    let params = weights.as_slice();
    let valid = params.len() == EXPECTED_PARAMS_LEN;

    if valid {
        let phase_base = ph * PARAMS_PER_PHASE;
        let mut raw: i32 = 0;

        if use_fast {
            let mut fi = 0usize;
            while fi < N_PATTERN_FEATURES {
                let pattern = FEATURE_TO_PATTERN[fi];
                let f = &EGEV2_FEATURE_TO_COORD[fi];
                let len = f.n_cells as usize;
                let abs_id = board.feat_id_abs[fi];
                let idx = abs_to_egev2_idx(side, len, abs_id);
                debug_assert!(idx < PATTERN_POW3[pattern]);
                let off = phase_base + PATTERN_OFFSETS[pattern] + idx;
                // Safety: `valid` ensures `params` has the full expected layout,
                // and `idx` is within the pattern table.
                raw += unsafe { *params.get_unchecked(off) } as i32;
                fi += 1;
            }
        } else {
            // Slow path: compute EGEV2 indices directly from bitboards.
            let mut fi = 0usize;
            while fi < N_PATTERN_FEATURES {
                let pattern = FEATURE_TO_PATTERN[fi];
                let f = &EGEV2_FEATURE_TO_COORD[fi];
                let idx = feature_idx_for_board(player, opp, f);
                debug_assert!(idx < PATTERN_POW3[pattern]);
                let off = phase_base + PATTERN_OFFSETS[pattern] + idx;
                raw += unsafe { *params.get_unchecked(off) } as i32;
                fi += 1;
            }
        }

        let num_player_discs = player.count_ones() as usize;
        debug_assert!(num_player_discs < EVAL_NUM_LEN);
        let off = phase_base + PATTERN_PARAMS_PER_PHASE + num_player_discs;
        raw += unsafe { *params.get_unchecked(off) } as i32;

        return round_clamp_raw_to_disc(raw);
    }

    // Safe fallback (missing / partial weights): treat missing params as 0.
    let mut raw: i32 = 0;
    if use_fast {
        let mut fi = 0;
        while fi < N_PATTERN_FEATURES {
            let pattern = FEATURE_TO_PATTERN[fi];
            let f = &EGEV2_FEATURE_TO_COORD[fi];
            let len = f.n_cells as usize;
            let abs_id = board.feat_id_abs[fi];
            let idx = abs_to_egev2_idx(side, len, abs_id);
            raw += weights.get_pattern_weight(ph, pattern, idx) as i32;
            fi += 1;
        }
    } else {
        let mut fi = 0;
        while fi < N_PATTERN_FEATURES {
            let pattern = FEATURE_TO_PATTERN[fi];
            let f = &EGEV2_FEATURE_TO_COORD[fi];
            let idx = feature_idx_for_board(player, opp, f);
            raw += weights.get_pattern_weight(ph, pattern, idx) as i32;
            fi += 1;
        }
    }

    let num_player_discs = player.count_ones() as usize;
    raw += weights.get_eval_num_weight(ph, num_player_discs) as i32;

    round_clamp_raw_to_disc(raw)
    }
}

// ---------------------------------------------------------------------------
// P2-1: Wasm SIMD (simd128) score_disc fast path
// ---------------------------------------------------------------------------

#[cfg(all(feature = "wasm_simd", target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn score_disc_wasm_simd(board: &Board, weights: &Weights) -> i32 {
    use core::arch::wasm32::*;

    let ph = phase(board.empty_count);

    let side = board.side;
    let player = board.player;
    let opp = board.opponent;

    // Fast path: incremental absolute pattern IDs are available.
    //
    // `feat_is_pattern_ids` is set by `recompute_features_in_place` and avoids
    // re-scanning the entire vector on every evaluation call.
    let use_fast = board.feat_is_pattern_ids && board.feat_id_abs.len() == N_PATTERN_FEATURES;

    // Same optimization strategy as the scalar path: when the weight buffer is
    // valid, index directly into the flat slice to avoid repeated bounds checks.
    let params = weights.as_slice();
    let valid = params.len() == EXPECTED_PARAMS_LEN;
    let phase_base = ph * PARAMS_PER_PHASE;

    // Accumulate in 4 lanes, then horizontal-sum at the end.
    // N_PATTERN_FEATURES is 64 in this engine, so it's a perfect multiple of 4.
    unsafe {
        let mut acc: v128 = i32x4_splat(0);

        let mut fi: usize = 0;
        while fi < N_PATTERN_FEATURES {
            // Unroll 4 features per iteration.
            let w0: i32;
            let w1: i32;
            let w2: i32;
            let w3: i32;

            if use_fast {
                let f0 = &EGEV2_FEATURE_TO_COORD[fi + 0];
                let f1 = &EGEV2_FEATURE_TO_COORD[fi + 1];
                let f2 = &EGEV2_FEATURE_TO_COORD[fi + 2];
                let f3 = &EGEV2_FEATURE_TO_COORD[fi + 3];

                let p0 = FEATURE_TO_PATTERN[fi + 0];
                let p1 = FEATURE_TO_PATTERN[fi + 1];
                let p2 = FEATURE_TO_PATTERN[fi + 2];
                let p3 = FEATURE_TO_PATTERN[fi + 3];

                let idx0 = abs_to_egev2_idx(side, f0.n_cells as usize, board.feat_id_abs[fi + 0]);
                let idx1 = abs_to_egev2_idx(side, f1.n_cells as usize, board.feat_id_abs[fi + 1]);
                let idx2 = abs_to_egev2_idx(side, f2.n_cells as usize, board.feat_id_abs[fi + 2]);
                let idx3 = abs_to_egev2_idx(side, f3.n_cells as usize, board.feat_id_abs[fi + 3]);

                if valid {
                    let o0 = phase_base + PATTERN_OFFSETS[p0] + idx0;
                    let o1 = phase_base + PATTERN_OFFSETS[p1] + idx1;
                    let o2 = phase_base + PATTERN_OFFSETS[p2] + idx2;
                    let o3 = phase_base + PATTERN_OFFSETS[p3] + idx3;
                    w0 = *params.get_unchecked(o0) as i32;
                    w1 = *params.get_unchecked(o1) as i32;
                    w2 = *params.get_unchecked(o2) as i32;
                    w3 = *params.get_unchecked(o3) as i32;
                } else {
                    w0 = weights.get_pattern_weight(ph, p0, idx0) as i32;
                    w1 = weights.get_pattern_weight(ph, p1, idx1) as i32;
                    w2 = weights.get_pattern_weight(ph, p2, idx2) as i32;
                    w3 = weights.get_pattern_weight(ph, p3, idx3) as i32;
                }
            } else {
                let f0 = &EGEV2_FEATURE_TO_COORD[fi + 0];
                let f1 = &EGEV2_FEATURE_TO_COORD[fi + 1];
                let f2 = &EGEV2_FEATURE_TO_COORD[fi + 2];
                let f3 = &EGEV2_FEATURE_TO_COORD[fi + 3];

                let p0 = FEATURE_TO_PATTERN[fi + 0];
                let p1 = FEATURE_TO_PATTERN[fi + 1];
                let p2 = FEATURE_TO_PATTERN[fi + 2];
                let p3 = FEATURE_TO_PATTERN[fi + 3];

                let idx0 = feature_idx_for_board(player, opp, f0);
                let idx1 = feature_idx_for_board(player, opp, f1);
                let idx2 = feature_idx_for_board(player, opp, f2);
                let idx3 = feature_idx_for_board(player, opp, f3);

                if valid {
                    let o0 = phase_base + PATTERN_OFFSETS[p0] + idx0;
                    let o1 = phase_base + PATTERN_OFFSETS[p1] + idx1;
                    let o2 = phase_base + PATTERN_OFFSETS[p2] + idx2;
                    let o3 = phase_base + PATTERN_OFFSETS[p3] + idx3;
                    w0 = *params.get_unchecked(o0) as i32;
                    w1 = *params.get_unchecked(o1) as i32;
                    w2 = *params.get_unchecked(o2) as i32;
                    w3 = *params.get_unchecked(o3) as i32;
                } else {
                    w0 = weights.get_pattern_weight(ph, p0, idx0) as i32;
                    w1 = weights.get_pattern_weight(ph, p1, idx1) as i32;
                    w2 = weights.get_pattern_weight(ph, p2, idx2) as i32;
                    w3 = weights.get_pattern_weight(ph, p3, idx3) as i32;
                }
            }

            acc = i32x4_add(acc, i32x4(w0, w1, w2, w3));
            fi += 4;
        }

        let mut raw: i32 =
            i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc);

        let num_player_discs = player.count_ones() as usize;
        if valid {
            let o = phase_base + PATTERN_PARAMS_PER_PHASE + num_player_discs;
            raw += *params.get_unchecked(o) as i32;
        } else {
            raw += weights.get_eval_num_weight(ph, num_player_discs) as i32;
        }

        round_clamp_raw_to_disc(raw)
    }
}


// ---------------------------------------------------------------------------
// Phase0 P0-2: tanh lookup table// ---------------------------------------------------------------------------
// Phase0 P0-2: tanh lookup table
// ---------------------------------------------------------------------------
//
// `evaluate()` maps the disc score (sd in [-64,64]) through `tanh(sd/50)` and
// scales it to "eval1000" in [-1000,1000] (actually smaller in practice).
//
// Calling `tanhf` at every node is expensive; we cache the 129 possible inputs
// once at runtime and then do a single array lookup.
//
// NOTE: We intentionally match the existing semantics exactly:
// `(tanhf(sd/50) * 1000.0) as i32` (float->int trunc toward zero).

const TANH_EVAL1000_LEN: usize = (SCORE_MAX as usize) * 2 + 1; // 129
static TANH_EVAL1000_LUT: OnceLock<[i16; TANH_EVAL1000_LEN]> = OnceLock::new();

#[inline(always)]
fn tanh_eval1000_from_disc(sd: i32) -> i32 {
    debug_assert!((-SCORE_MAX..=SCORE_MAX).contains(&sd));
    let idx = (sd + SCORE_MAX) as usize;

    let lut = TANH_EVAL1000_LUT.get_or_init(|| {
        let mut lut = [0i16; TANH_EVAL1000_LEN];
        let mut s = -SCORE_MAX;
        while s <= SCORE_MAX {
            let y = libm::tanhf((s as f32) / 50.0);
            lut[(s + SCORE_MAX) as usize] = (y * 1000.0) as i16;
            s += 1;
        }
        lut
    });

    lut[idx] as i32
}

/// P0-3: Proactively initialize the large OnceLock evaluation tables.
///
/// This avoids a noticeable one-time stall on the first evaluation (especially
/// in Wasm where the UI thread is also the search thread).
#[inline]
pub fn warm_up_eval_tables() {
    // abs_id -> egev2_id mapping tables (largest one-time allocation in eval).
    let _ = abs_to_egev2_tables();

    // tanh LUT used by the search-facing evaluate() function.
    let _ = tanh_eval1000_from_disc(0);
}

/// Search-facing evaluation output (tanh-mapped, [-1000, 1000]).
///
/// Kept for compatibility with Sonetto's existing search interface.
#[inline(always)]
pub fn evaluate(board: &Board, weights: &Weights, _feats: &FeatureDefs, _swap: &SwapTables) -> i32 {
    let sd = score_disc(board, weights);
    tanh_eval1000_from_disc(sd)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::coord::{Move, PASS};
    use crate::features::update::recompute_features_in_place;
    use crate::movegen::{legal_moves, push_moves_from_mask};

    #[inline(always)]
    fn rng_next(state: &mut u64) -> u64 {
        // xorshift64* (deterministic, no external deps)
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545F4914F6CDD1Du64)
    }

    #[test]
    fn tanh_eval1000_lut_matches_libm_cast() {
        for sd in -SCORE_MAX..=SCORE_MAX {
            let y = libm::tanhf(sd as f32 / 50.0);
            let expected = (y * 1000.0) as i32;
            assert_eq!(tanh_eval1000_from_disc(sd), expected, "sd={sd}");
        }
    }

    use crate::egev2::decode_egev2;

    fn parse_board_str(s: &str) -> Board {
        let s = s.split_whitespace().collect::<String>();
        assert_eq!(s.len(), 65, "board string must be 64 cells + side char");
        let bytes = s.as_bytes();

        let mut black: u64 = 0;
        let mut white: u64 = 0;
        for i in 0..64 {
            match bytes[i] {
                b'X' | b'x' => black |= 1u64 << i,
                b'O' | b'o' => white |= 1u64 << i,
                b'-' | b'.' => {}
                other => panic!("invalid board char at {i}: {other}")
            }
        }
        let side = match bytes[64] {
            b'X' | b'x' => Color::Black,
            b'O' | b'o' => Color::White,
            other => panic!("invalid side char: {other}"),
        };
        let occupied = black | white;
        let empty_count = 64u8 - occupied.count_ones() as u8;

        let (player, opponent) = if side == Color::Black { (black, white) } else { (white, black) };

        Board {
            player,
            opponent,
            side,
            empty_count,
            hash: 0,
            feat_is_pattern_ids: false,
            feat_id_abs: Vec::new(),
        }
    }

    #[test]
    fn score_disc_is_zero_when_weights_missing() {
        // Weights::zeroed() produces an empty param vector.
        // By design, missing parameters read as 0, so the raw sum is 0 and the
        // disc score should always be 0 regardless of board.
        let weights = Weights::zeroed();

        let cases: [&str; 5] = [
            "---------------------------OX------XO---------------------------X",
            "---------------------------XO------OX-------OOO-----------------X",
            "---------------------------XO------OX-------OXO-------O-------O-X",
            "---------------------------OX------XXX------XXX-----OOO------OOOX",
            "--------------------------XXX-O----XXO-----OOOOO--OOOOOO-OOOOOOOX",
        ];

        for b in cases {
            let board = parse_board_str(b);
            let got = score_disc(&board, &weights);
            assert_eq!(got, 0, "board={b}");
        }
    }

    #[test]
    fn p1_2_static_occ_map_matches_dynamic_builder() {
        // Build the OccMap dynamically (the pre-P1-2 way) and compare against
        // the compile-time generated mapping.
        let mut squares_rev: Vec<Vec<u8>> = Vec::with_capacity(N_PATTERN_FEATURES);
        for fi in 0..N_PATTERN_FEATURES {
            let f = &EGEV2_FEATURE_TO_COORD[fi];
            let n = f.n_cells as usize;
            let mut v: Vec<u8> = Vec::with_capacity(n);
            let mut j = n;
            while j > 0 {
                j -= 1;
                v.push(f.cells[j]);
            }
            squares_rev.push(v);
        }

        let square_slices: Vec<&[u8]> = squares_rev.iter().map(|v| v.as_slice()).collect();
        let occ_dyn = OccMap::build_from_feature_squares(&square_slices);

        let (_feats, occ_static) = build_sonetto_feature_defs_and_occ();
        assert_eq!(occ_static.total_occurrences(), EGEV2_TOTAL_OCC);

        for sq in 0u8..64u8 {
            assert_eq!(occ_static.occ_for_sq(sq), occ_dyn.occ_for_sq(sq), "sq={sq}");
        }
    }

    #[test]
    fn p0_3_abs_to_egev2_mapping_matches_slow_idx_random_positions() {
        let (_feats, occ) = build_sonetto_feature_defs_and_occ();

        let mut rng: u64 = 0x1234_5678_9abc_def0;
        for _ in 0..200 {
            let r1 = rng_next(&mut rng);
            let r2 = rng_next(&mut rng);

            // Random but consistent bitboards (no overlap).
            let occ_mask = r1;
            let black = r2 & occ_mask;
            let white = occ_mask & !black;

            let side = if (rng_next(&mut rng) & 1) == 0 { Color::Black } else { Color::White };
            let empty_count = 64u8 - (black | white).count_ones() as u8;

            let (player, opponent) = if side == Color::Black { (black, white) } else { (white, black) };

            let mut board = Board {
                player,
                opponent,
                side,
                empty_count,
                hash: 0,
                feat_is_pattern_ids: false,
                feat_id_abs: vec![0u16; N_PATTERN_FEATURES],
            };
            recompute_features_in_place(&mut board, &occ);

            let player = board.player;
            let opp = board.opponent;

            for fi in 0..N_PATTERN_FEATURES {
                let f = &EGEV2_FEATURE_TO_COORD[fi];
                let len = f.n_cells as usize;
                let slow = feature_idx_for_board(player, opp, f);
                let fast = abs_to_egev2_idx(board.side, len, board.feat_id_abs[fi]);
                assert_eq!(slow, fast, "fi={fi} len={len} side={:?}", board.side);
            }
        }
    }

    #[test]
    fn p0_3_incremental_pattern_ids_match_recompute_during_random_playout() {
        let (_feats, occ) = build_sonetto_feature_defs_and_occ();

        let mut board = Board::new_start(N_PATTERN_FEATURES);
        recompute_features_in_place(&mut board, &occ);

        let mut rng: u64 = 0x0ddc_0ffe_e0dd_f00d;
        let mut undos: Vec<crate::board::Undo> = Vec::new();

        for _ply in 0..40 {
            let me = board.player;
            let opp = board.opponent;

            let mask = legal_moves(me, opp);
            let mv: Move = if mask == 0 {
                PASS
            } else {
                let mut buf = [PASS; 64];
                let n = push_moves_from_mask(mask, &mut buf) as usize;
                buf[(rng_next(&mut rng) as usize) % n]
            };

            let mut u = crate::board::Undo::default();
            assert!(board.apply_move_with_occ(mv, &mut u, Some(&occ)));

            let mut check = board.clone();
            recompute_features_in_place(&mut check, &occ);
            assert_eq!(board.feat_id_abs, check.feat_id_abs, "after mv={mv}");

            undos.push(u);
        }

        while let Some(u) = undos.pop() {
            board.undo_move_with_occ(&u, Some(&occ));

            let mut check = board.clone();
            recompute_features_in_place(&mut check, &occ);
            assert_eq!(board.feat_id_abs, check.feat_id_abs, "after undo mv={}", u.mv);
        }
    }

}
