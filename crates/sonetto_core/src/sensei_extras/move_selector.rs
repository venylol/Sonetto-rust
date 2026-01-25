//! Sensei-style move selection with a controllable expected error.
//!
//! This is a Rust port of:
//! `sensei_engine/engine/playagainstsensei/move_selector.{h,cpp}`.
//!
//! # Key idea
//!
//! Given a set of candidate moves with evaluations `eval[i]`, define the
//! *relative error* `err[i] = eval[i] - best_eval`, where **smaller eval is
//! considered better** (this matches Sensei's `Node::GetEval()` convention).
//!
//! We then assign probabilities:
//!
//! ```text
//! p[i] ∝ exp(-λ * err[i])
//! ```
//!
//! and choose `λ` (via binary search) so that `E[err]` matches a caller-provided
//! `expected_error`.
//!
//! - `λ = 0` => uniform random among included moves.
//! - `λ → ∞` => always choose the best move.
//!
//! # Notes
//!
//! - This module is **not** used by Sonetto's default searcher. It is meant as an
//!   optional helper for UI/"play against engine" strength shaping.
//! - If your evaluations are "higher is better", pass `-score` as `eval`.

use core::cell::RefCell;

/// Ported from Sensei's `kErrorPercentage` table (length = 60).
///
/// The entries sum to ~1.0; Sensei multiplies by 2 to obtain a per-side total
/// error allocation that sums to 2.0.
const ERROR_PERCENTAGE: [f64; 60] = [
    0.0,
    0.0013976365946040002,
    0.003801571230374871,
    0.003271857745814194,
    0.008279673683894071,
    0.008448142395832598,
    0.007552751937430454,
    0.009881970091850992,
    0.008117337597219946,
    0.010962527491184877,
    0.008174796896444377,
    0.011785666258739507,
    0.011155759522086578,
    0.013035210812129754,
    0.012621659877825714,
    0.014082477740390948,
    0.013300534179687138,
    0.016098406205464023,
    0.0159153368478561,
    0.015072612919671647,
    0.017690716977093606,
    0.016949190538568137,
    0.016673427732177476,
    0.01835123305537761,
    0.019839462019340278,
    0.019490903403816973,
    0.02057108925096853,
    0.02083143089724683,
    0.01944704940037023,
    0.021583527522205877,
    0.022466329982804203,
    0.020537626724224736,
    0.02414092384423285,
    0.0217643074362023,
    0.026820380927631608,
    0.024126893706231237,
    0.027589297421430747,
    0.024389761884498425,
    0.028485842029613167,
    0.02738794538885042,
    0.028161200889229937,
    0.02987614383696257,
    0.029009659154296107,
    0.02963863278982487,
    0.028527184060206438,
    0.028527184060206438,
    0.025580160913490895,
    0.02408138914173271,
    0.022178454195567812,
    0.023609865438258222,
    0.01987135607499622,
    0.018439944832305814,
    0.01613284671173422,
    0.013168983432751732,
    0.011501810338324083,
    0.008251664810803515,
    0.005725644970761622,
    0.0038227100245967305,
    0.0018018941525632167,
    0.0,
];

/// Returns how much of the total game error should happen when playing this move.
///
/// This matches Sensei's `GetMoveMultiplier`:
/// it returns `ERROR_PERCENTAGE[move] * 2` so that summing over moves `0..60`
/// yields `2.0`.
#[inline]
pub fn get_move_multiplier(mv: usize) -> f64 {
    ERROR_PERCENTAGE.get(mv).copied().unwrap_or(0.0) * 2.0
}

// -----------------------------------------------------------------------------
// RNG (no `rand` dependency)
// -----------------------------------------------------------------------------

/// A tiny RNG trait used by the move selector.
///
/// This keeps `sonetto_core` dependency-free (no `rand` crate), and makes tests
/// deterministic.
pub trait Rng64 {
    /// Next random 64-bit value.
    fn next_u64(&mut self) -> u64;

    /// Uniform in `[0, 1)` using the top 53 bits.
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // IEEE-754 f64 mantissa is 52 bits (+ hidden bit). Using 53 bits gives a
        // nicely distributed fraction.
        let x = self.next_u64() >> 11;
        (x as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform integer in `0..upper`.
    #[inline]
    fn gen_usize(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % (upper as u64)) as usize
    }
}

/// Deterministic xorshift64 RNG.
///
/// This is *not* cryptographically secure; it's a simple helper for selection.
#[derive(Clone, Copy, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        // Avoid the all-zero lockup state.
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }
}

impl Rng64 for XorShift64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

thread_local! {
    // A deterministic default seed (matches the "no srand()" spirit of C rand()).
    static TLS_RNG: RefCell<XorShift64> = RefCell::new(XorShift64::new(1));
}

// -----------------------------------------------------------------------------
// Core algorithm (ported from Sensei)
// -----------------------------------------------------------------------------

#[inline]
fn weight(error: f64, lambda: f64) -> f64 {
    // Sensei uses `exp(-lambda * error)`.
    (-lambda * error).exp()
}

#[inline]
fn total_weight(errors: &[f64], lambda: f64) -> f64 {
    let mut total = 0.0;
    for &e in errors {
        total += weight(e, lambda);
    }
    total
}

#[inline]
fn expected_error(errors: &[f64], lambda: f64) -> f64 {
    let mut w_times_e = 0.0;
    for &e in errors {
        w_times_e += e * weight(e, lambda);
    }
    w_times_e / total_weight(errors, lambda)
}

/// Select the next move index from a list of evaluations.
///
/// - `evals[i]` is the evaluation of move `i`.
/// - **Smaller** eval is treated as better (Sensei convention).
/// - `expected_error` is the desired expectation of `(eval - best_eval)`.
/// - Moves with error greater than `max(max_error, 0.01)` are excluded.
///
/// Returns `None` when `evals` is empty.
///
/// This is the direct port of Sensei's `FindNextMove` logic, except it returns an
/// index instead of a pointer.
pub fn find_next_move_index_with_rng(
    evals: &[f64],
    expected_error_target: f64,
    max_error: f64,
    rng: &mut impl Rng64,
) -> Option<usize> {
    if evals.is_empty() {
        return None;
    }

    // Find the best (minimum) evaluation.
    let mut best_eval = f64::INFINITY;
    let mut best_idx = 0usize;
    for (i, &ev) in evals.iter().enumerate() {
        if ev.is_finite() && ev < best_eval {
            best_eval = ev;
            best_idx = i;
        }
    }

    // If all evals were NaN/inf, just return the first move.
    if !best_eval.is_finite() {
        return Some(0);
    }

    let threshold = max_error.max(0.01);

    let mut errors: Vec<f64> = Vec::with_capacity(evals.len());
    let mut idx_for_error: Vec<usize> = Vec::with_capacity(evals.len());

    for (i, &ev) in evals.iter().enumerate() {
        let err = ev - best_eval;
        if err.is_finite() && err <= threshold {
            errors.push(err);
            idx_for_error.push(i);
        }
    }

    // Defensive fallback (shouldn't happen unless all evals are non-finite).
    if errors.is_empty() {
        return Some(best_idx);
    }

    let mut target = expected_error_target;
    if !target.is_finite() {
        target = 0.0;
    }
    if target < 0.0 {
        target = 0.0;
    }

    // Binary search for a lambda that matches the target expected error.
    let mut l = 0.0f64; // all moves same probability
    let mut u = 100.0f64; // exp(-100*0.01) ~ 1/e

    let err_l = expected_error(&errors, l);
    if err_l <= target {
        // Can't achieve such a large expected error: return a uniform random move.
        let j = rng.gen_usize(idx_for_error.len());
        return Some(idx_for_error[j]);
    }

    let err_u = expected_error(&errors, u);
    if err_u >= target {
        // Can't achieve such a small expected error: just pick the best move.
        return Some(best_idx);
    }

    while u - l > 1e-2 {
        let mid = (l + u) * 0.5;
        let err_mid = expected_error(&errors, mid);
        if err_mid < target {
            u = mid;
        } else {
            l = mid;
        }
    }

    let lambda = (l + u) * 0.5;
    let mut point = rng.next_f64() * total_weight(&errors, lambda);
    for (k, &err) in errors.iter().enumerate() {
        point -= weight(err, lambda);
        if point <= 0.0 {
            return Some(idx_for_error[k]);
        }
    }

    // Floating point edge case fallback.
    Some(idx_for_error[0])
}

/// Convenience wrapper that uses a deterministic thread-local RNG.
#[inline]
pub fn find_next_move_index(evals: &[f64], expected_error: f64, max_error: f64) -> Option<usize> {
    TLS_RNG.with(|cell| {
        find_next_move_index_with_rng(evals, expected_error, max_error, &mut *cell.borrow_mut())
    })
}

/// Same as [`find_next_move_index_with_rng`], but interprets `total_error` as the
/// desired *end-of-game* error and allocates a per-move expected error using
/// Sensei's `GetMoveMultiplier` schedule.
///
/// - `n_empties` should be the number of empty squares *before* playing the move.
/// - The move number is computed as `max(0, 59 - n_empties)`.
pub fn find_next_move_total_error_index_with_rng(
    evals: &[f64],
    n_empties: u8,
    total_error: f64,
    max_error: f64,
    rng: &mut impl Rng64,
) -> Option<usize> {
    if evals.is_empty() {
        return None;
    }

    let mv_idx = (59i32 - (n_empties as i32)).max(0) as usize;
    let expected_error = total_error * get_move_multiplier(mv_idx);

    find_next_move_index_with_rng(evals, expected_error, max_error, rng)
}

/// Convenience wrapper that uses a deterministic thread-local RNG.
#[inline]
pub fn find_next_move_total_error_index(
    evals: &[f64],
    n_empties: u8,
    total_error: f64,
    max_error: f64,
) -> Option<usize> {
    TLS_RNG.with(|cell| {
        find_next_move_total_error_index_with_rng(
            evals,
            n_empties,
            total_error,
            max_error,
            &mut *cell.borrow_mut(),
        )
    })
}
