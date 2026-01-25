//! Sensei-style win probability tables.
//!
//! This is a Rust port of:
//! `sensei_engine/engine/estimators/win_probability.{h,cpp}`.
//!
//! The original code builds a precomputed table that maps:
//!
//! - search depth in `[1,4]`
//! - empty squares in `[0,63]`
//! - `delta = goal - eval` in "Sensei eval-large" units (1/8 disc)
//!
//! to a compressed probability byte `0..=255`.
//!
//! # Important
//!
//! These APIs are **optional** helpers and are not used by Sonetto's default
//! [`crate::search::Searcher`] path.

use std::sync::OnceLock;

/// Compressed probability byte (0..=255).
pub type Probability = u8;

/// Sensei uses 255 as the scale for probability bytes.
pub const PROB_STEP: Probability = 255;

/// Sensei's evaluation range in `Eval` units.
pub const MIN_EVAL: i32 = -64;
pub const MAX_EVAL: i32 = 64;

/// Sensei's `EvalLarge` is `Eval * 8`.
pub const MIN_EVAL_LARGE: i32 = MIN_EVAL * 8;
pub const MAX_EVAL_LARGE: i32 = MAX_EVAL * 8;

/// Maximum (exclusive) offset used by the lookup table.
///
/// Matches Sensei's `kMaxCDFOffset`.
pub const MAX_CDF_OFFSET: usize = 4 * 64 * (256 * 8 + 1);

/// Ported from Sensei's `kErrors` table.
///
/// `ERRORS[depth][empties]` is a depth-dependent empirical stddev estimate (in
/// discs) used to map eval deltas to win probability.
///
/// Depth is indexed by `1..=4`. Row `0` exists only for alignment.
pub const ERRORS: [[f32; 60]; 5] = [
    [0.0; 60],
    [
        2.00, 2.00, 2.00, 2.00, 6.64, 6.87, 7.64, 7.77, 8.18, 8.30, 8.72, 8.73, 8.98, 8.71,
        8.65, 8.35, 8.29, 8.05, 8.20, 7.64, 7.55, 7.05, 6.82, 6.00, 6.36, 5.61, 5.86, 5.19,
        5.76, 5.13, 5.58, 4.91, 5.19, 4.39, 4.89, 4.14, 4.82, 4.03, 4.38, 3.83, 4.16, 3.52,
        3.82, 3.20, 3.28, 2.79, 2.96, 2.41, 2.84, 2.57, 2.51, 2.00, 2.00, 2.00, 2.00, 2.00,
        2.00, 2.00, 2.00, 2.00,
    ],
    [
        2.00, 2.00, 2.00, 2.00, 5.57, 5.86, 6.40, 7.08, 7.23, 7.63, 7.77, 8.06, 8.03, 8.19,
        7.91, 7.81, 7.57, 7.51, 7.41, 7.29, 6.92, 6.56, 6.18, 5.41, 5.67, 5.14, 5.13, 4.66,
        4.98, 4.56, 4.78, 4.37, 4.21, 3.90, 4.03, 3.73, 3.97, 3.63, 3.54, 3.46, 3.51, 3.15,
        3.17, 2.94, 2.67, 2.52, 2.56, 2.21, 2.10, 2.48, 2.09, 2.00, 2.00, 2.00, 2.00, 2.00,
        2.00, 2.00, 2.00, 2.00,
    ],
    [
        2.00, 2.00, 2.00, 2.00, 5.17, 4.96, 5.63, 5.93, 6.68, 6.74, 7.21, 7.25, 7.47, 7.38,
        7.51, 7.21, 7.10, 6.97, 6.89, 6.66, 6.57, 6.03, 5.75, 5.01, 5.30, 4.71, 5.16, 4.30,
        4.73, 4.09, 4.59, 3.88, 4.40, 3.50, 3.97, 3.29, 4.03, 3.17, 3.81, 3.09, 3.50, 2.74,
        3.25, 2.59, 3.00, 2.28, 2.61, 2.05, 2.25, 2.00, 2.37, 2.00, 2.00, 2.00, 2.00, 2.00,
        2.00, 2.00, 2.00, 2.00,
    ],
    [
        2.00, 2.00, 2.00, 2.00, 2.00, 4.72, 4.71, 5.25, 5.53, 6.34, 6.37, 6.77, 6.71, 6.87,
        6.81, 7.03, 6.56, 6.54, 6.37, 6.27, 6.05, 5.93, 5.32, 4.77, 4.93, 4.39, 4.64, 3.98,
        4.05, 3.66, 3.96, 3.53, 3.67, 3.24, 3.15, 2.96, 3.34, 2.93, 3.06, 2.86, 2.80, 2.57,
        2.72, 2.40, 2.50, 2.13, 2.13, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
        2.00, 2.00, 2.00, 2.00,
    ],
];

/// Convert `(depth, empties, eval_delta)` into the lookup table offset.
///
/// Matches Sensei's `DataToCDFOffset`.
#[inline]
pub fn data_to_cdf_offset(depth: u8, n_empties: u8, eval_delta: i32) -> usize {
    debug_assert!((1..=4).contains(&depth));
    debug_assert!(n_empties <= 63);
    debug_assert!(eval_delta >= 2 * MIN_EVAL_LARGE && eval_delta <= -2 * MIN_EVAL_LARGE);

    let depth_part = (depth as usize).wrapping_sub(1);
    let empties_part = (n_empties as usize) << 2;
    let delta_part = ((eval_delta - 2 * MIN_EVAL_LARGE) as usize) << 8;
    depth_part | empties_part | delta_part
}

/// Inverse of [`data_to_cdf_offset`].
///
/// Matches Sensei's `CDFOffsetToDepthEmptiesEval`.
#[inline]
pub fn cdf_offset_to_depth_empties_eval(offset: usize) -> (u8, u8, i32) {
    let depth = ((offset & 3) + 1) as u8;
    let empties = ((offset >> 2) & 63) as u8;
    let eval_delta = (offset >> 8) as i32 + 2 * MIN_EVAL_LARGE;
    (depth, empties, eval_delta)
}

// -----------------------------------------------------------------------------
// Probability rescaling (ported from Sensei)
// -----------------------------------------------------------------------------

#[inline]
fn base_rescale_prob(x: f64) -> f64 {
    if x < f64::MIN_POSITIVE {
        return 0.0;
    }
    // Sensei: pow(-log(x) + 10, -3.5)
    (-x.ln() + 10.0).powf(-3.5)
}

#[inline]
fn rescale_prob(x: f64) -> f64 {
    if x <= 1e-14 {
        return 0.0;
    }
    if x >= 1.0 - 1e-14 {
        return 1.0;
    }

    let num = base_rescale_prob(x) - base_rescale_prob(1.0 - x);
    let den = base_rescale_prob(1.0) - base_rescale_prob(0.0);
    (num / den) * 0.5 + 0.5
}

#[inline]
fn inverse<F: Fn(f64) -> f64>(f: F, y: f64, mut l: f64, mut u: f64) -> f64 {
    debug_assert!(f(l) <= y && y <= f(u));

    if f(l) == y {
        return l;
    }
    if f(u) == y {
        return u;
    }

    while u - l > 1e-14 {
        let mid = (l + u) * 0.5;
        let fmid = f(mid);
        if fmid == y {
            return mid;
        } else if fmid < y {
            l = mid;
        } else {
            u = mid;
        }
    }

    (l + u) * 0.5
}

/// Inverse of [`rescale_prob`].
#[inline]
pub fn inverse_rescale_prob(y: f64) -> f64 {
    if y <= 1e-14 {
        return 0.0;
    }
    if y >= 1.0 - 1e-14 {
        return 1.0;
    }
    inverse(rescale_prob, y, 0.0, 1.0)
}

/// Convert a probability in `[0,1]` to a compressed byte.
///
/// Matches Sensei's `ProbabilityToByteExplicit`.
#[inline]
pub fn probability_to_byte_explicit(probability: f64) -> Probability {
    debug_assert!((0.0..=1.0).contains(&probability));
    let byte = (rescale_prob(probability) * (PROB_STEP as f64)).round();
    byte.clamp(0.0, PROB_STEP as f64) as u8
}

/// Convert a compressed byte to an explicit probability in `[0,1]`.
///
/// Matches Sensei's `ByteToProbabilityExplicit`.
#[inline]
pub fn byte_to_probability_explicit(byte: Probability) -> f64 {
    inverse_rescale_prob(byte as f64 / (PROB_STEP as f64))
}

// -----------------------------------------------------------------------------
// Gaussian CDF helpers
// -----------------------------------------------------------------------------

/// Complementary error function approximation.
///
/// Rust's standard library does not expose `erf/erfc`, and we avoid relying on
/// platform-specific libc symbols for WASM portability.
///
/// This implementation is the classic Numerical Recipes approximation
/// (roughly 1e-7 absolute error).
#[inline]
fn erfc(x: f64) -> f64 {
    // Numerical Recipes in C: `erfcc`.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z
            - 1.26551223
            + t
                * (1.00002368
                    + t
                        * (0.37409196
                            + t
                                * (0.09678418
                                    + t
                                        * (-0.18628806
                                            + t
                                                * (0.27886807
                                                    + t
                                                        * (-1.13520398
                                                            + t
                                                                * (1.48851587
                                                                    + t
                                                                        * (-0.82215223
                                                                            + t
                                                                                * 0.17087277)))))))))
            .exp();

    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Standard normal CDF using the complementary error function.
///
/// Matches Sensei's `GaussianCDF(double value)`.
#[inline]
pub fn gaussian_cdf_standard(value: f64) -> f64 {
    // 0.70710678118 ~= 1/sqrt(2)
    0.5 * erfc(-value * 0.70710678118)
}

/// Gaussian CDF `P(X <= x)` for `X ~ N(mean, stddev)`.
///
/// Matches Sensei's `GaussianCDF(double x, double mean, double stddev)`.
#[inline]
pub fn gaussian_cdf(x: f64, mean: f64, stddev: f64) -> f64 {
    gaussian_cdf_standard((x - mean) / stddev)
}

/// Uncompressed probability estimate.
///
/// Matches Sensei's `ProbabilityExplicit`.
#[inline]
pub fn probability_explicit(depth: u8, empties: u8, delta: i32) -> f64 {
    debug_assert!((1..=4).contains(&depth));
    debug_assert!(empties <= 63);

    let error = if empties < 60 {
        // Sensei uses max(3, kErrors[depth][empties]).
        (ERRORS[depth as usize][empties as usize]).max(3.0)
    } else {
        3.0
    };

    // Sensei uses stddev = 8 * error (EvalLarge units).
    1.0 - gaussian_cdf(delta as f64, 0.0, 8.0 * (error as f64))
}

// -----------------------------------------------------------------------------
// Precomputed tables
// -----------------------------------------------------------------------------

struct WinProbabilityData {
    win_probability: Box<[Probability]>,
    byte_to_probability: [f64; (PROB_STEP as usize) + 1],
}

impl WinProbabilityData {
    fn new() -> Self {
        let mut win_probability = vec![0u8; MAX_CDF_OFFSET + 1].into_boxed_slice();

        for depth in 1u8..=4 {
            for empties in 0u8..64 {
                for delta in (2 * MIN_EVAL_LARGE)..=(-2 * MIN_EVAL_LARGE) {
                    let prob = probability_explicit(depth, empties, delta);
                    let byte = probability_to_byte_explicit(prob);
                    let off = data_to_cdf_offset(depth, empties, delta);
                    win_probability[off] = byte;
                }
            }
        }

        let mut byte_to_probability = [0.0f64; (PROB_STEP as usize) + 1];
        for i in 0u16..=(PROB_STEP as u16) {
            byte_to_probability[i as usize] = byte_to_probability_explicit(i as u8);
        }

        Self {
            win_probability,
            byte_to_probability,
        }
    }
}

static DATA: OnceLock<WinProbabilityData> = OnceLock::new();

#[inline]
fn data() -> &'static WinProbabilityData {
    DATA.get_or_init(WinProbabilityData::new)
}

/// Convert a compressed byte to a probability using the precomputed inverse table.
///
/// Matches Sensei's `ByteToProbability(Probability byte)`.
#[inline]
pub fn byte_to_probability(byte: Probability) -> f64 {
    data().byte_to_probability[byte as usize]
}

/// Return the compressed win probability byte for a given `(depth, empties, goal, eval)`.
///
/// Matches Sensei's `WinProbability(depth, n_empties, goal, eval)`.
#[inline]
pub fn win_probability(depth: u8, n_empties: u8, goal: i32, eval: i32) -> Probability {
    let delta = goal - eval;
    let off = data_to_cdf_offset(depth, n_empties, delta);
    data().win_probability[off]
}
