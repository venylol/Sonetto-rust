//! Sensei-style proof/disproof time estimators.
//!
//! This is a direct port of Sensei's `endgame_time.h` regression models.
//! It is primarily used for Sensei's disproof-number move ordering in the
//! endgame, via `disproof_number_over_prob`.
//!
//! Notes on units / quantization
//! - Sensei evaluates positions in `EvalLarge` units (1/8 disc).
//! - The estimators take an `eval_delta` in **disc units** in `[-128, 128]`,
//!   obtained by `(approx_eval - lower) >> 3`.
//!
//! In Sonetto, we keep the same interface: pass `lower` and `approx_eval` in
//! `EvalLarge` (1/8 disc) and this module performs the same `>> 3` quantization.

#![allow(clippy::excessive_precision)]

use std::sync::OnceLock;

use crate::movegen::legal_moves;
use crate::sensei_extras::win_probability::probability_explicit;

// --- Proof-number encoding (Sensei compatible) ---

const MAX_PROOF_NUMBER: f64 = 1.0e25;
const PROOF_NUMBER_STEP: i32 = 255;
const PROOF_OFFSET: f64 = 1.99;

// In Sensei:
//   constexpr double kBaseLogProofNumber = pow(kMaxProofNumber, 1.0 / (kProofNumberStep - kProofOffset));
static BASE_LOG_PROOF_NUMBER: OnceLock<f64> = OnceLock::new();

#[inline(always)]
fn base_log_proof_number() -> f64 {
    *BASE_LOG_PROOF_NUMBER.get_or_init(|| {
        MAX_PROOF_NUMBER.powf(1.0 / ((PROOF_NUMBER_STEP as f64) - PROOF_OFFSET))
    })
}

#[inline(always)]
fn bound(x: f64) -> f64 {
    x.max(1.0).min(MAX_PROOF_NUMBER * 0.99)
}

#[inline(always)]
fn proof_number_to_byte(proof_number: f64) -> u8 {
    // Sensei:
    //   int byte = round(log(proof_number) / log(kBaseLogProofNumber) + kProofOffset);
    //   return clamp(byte, 0, kProofNumberStep);
    let proof_number = bound(proof_number);
    let byte = (proof_number.ln() / base_log_proof_number().ln() + PROOF_OFFSET).round() as i32;
    byte.clamp(0, PROOF_NUMBER_STEP) as u8
}

#[inline(always)]
fn byte_to_proof_number_explicit(byte: u8) -> f64 {
    // Sensei:
    //   Bound(pow(kBaseLogProofNumber, byte - kProofOffset));
    bound(base_log_proof_number().powf((byte as f64) - PROOF_OFFSET))
}

// --- Table indexing (Sensei compatible) ---

const PROOF_TABLE_LEN: usize = 64 * 16 * 257; // empties * moves(0..15) * eval_delta(-128..128)

#[inline(always)]
fn data_to_proof_number_offset(empties: u8, n_moves: u8, eval_delta: i32) -> usize {
    debug_assert!(empties < 64);
    debug_assert!(n_moves <= 15);
    debug_assert!((-128..=128).contains(&eval_delta));

    // Sensei uses eval_delta - 2*kMinEval where kMinEval = -64, so shift by +128.
    (empties as usize) | ((n_moves as usize) << 6) | (((eval_delta + 128) as usize) << 10)
}

// --- Regression models (ported verbatim) ---

#[inline(always)]
fn log_moves(moves_player: u8) -> f64 {
    // Sensei: log(max(moves_player, 1) + 2)
    ((moves_player.max(1) as f64) + 2.0).ln()
}

#[inline(always)]
fn clamp_error(error: i32) -> f64 {
    // Sensei: error clamped to [-70*8, 70*8]
    let e = error.clamp(-70 * 8, 70 * 8);
    e as f64
}

#[inline]
fn log_proof_number(n_empties: u8, moves_player: u8, error: i32) -> f64 {
    let log_moves = log_moves(moves_player);
    let e = clamp_error(error);

    match n_empties {
        0 => 0.0290 + 0.9975 * log_moves + 0.0171 * e,
        1 => -0.2053 + 1.0401 * log_moves + 0.0266 * e,
        2 => -0.4111 + 1.0678 * log_moves + 0.0332 * e,
        3 => -0.5232 + 1.0743 * log_moves + 0.0365 * e,
        4 => -0.5227 + 1.0556 * log_moves + 0.0371 * e,
        5 => -0.4936 + 1.0359 * log_moves + 0.0381 * e,
        6 => -0.4751 + 1.0235 * log_moves + 0.0392 * e,
        7 => -0.4872 + 1.0253 * log_moves + 0.0400 * e,
        8 => -0.5159 + 1.0291 * log_moves + 0.0403 * e,
        9 => -0.5440 + 1.0312 * log_moves + 0.0405 * e,
        10 => -0.5690 + 1.0338 * log_moves + 0.0405 * e,
        11 => -0.5894 + 1.0351 * log_moves + 0.0407 * e,
        12 => -0.6071 + 1.0370 * log_moves + 0.0410 * e,
        13 => -0.6234 + 1.0382 * log_moves + 0.0412 * e,
        14 => -0.6402 + 1.0400 * log_moves + 0.0414 * e,
        15 => -0.6558 + 1.0409 * log_moves + 0.0416 * e,
        16 => -0.6691 + 1.0414 * log_moves + 0.0418 * e,
        17 => -0.6794 + 1.0416 * log_moves + 0.0419 * e,
        18 => -0.6871 + 1.0416 * log_moves + 0.0419 * e,
        19 => -0.6936 + 1.0416 * log_moves + 0.0419 * e,
        20 => -0.6990 + 1.0416 * log_moves + 0.0420 * e,
        21 => -0.7039 + 1.0416 * log_moves + 0.0420 * e,
        22 => -0.7086 + 1.0416 * log_moves + 0.0420 * e,
        23 => -0.7134 + 1.0416 * log_moves + 0.0420 * e,
        24 => -0.7186 + 1.0417 * log_moves + 0.0420 * e,
        25 => -0.7245 + 1.0419 * log_moves + 0.0420 * e,
        26 => -0.7316 + 1.0421 * log_moves + 0.0420 * e,
        27 => -0.7403 + 1.0425 * log_moves + 0.0420 * e,
        28 => -0.7510 + 1.0432 * log_moves + 0.0419 * e,
        29 => -0.7639 + 1.0440 * log_moves + 0.0418 * e,
        30 => -0.7795 + 1.0452 * log_moves + 0.0416 * e,
        31 => -0.7983 + 1.0466 * log_moves + 0.0414 * e,
        32 => -0.8205 + 1.0482 * log_moves + 0.0410 * e,
        33 => -0.8467 + 1.0504 * log_moves + 0.0407 * e,
        34 => -0.8773 + 1.0530 * log_moves + 0.0403 * e,
        35 => -0.9131 + 1.0564 * log_moves + 0.0397 * e,
        36 => -0.9550 + 1.0610 * log_moves + 0.0389 * e,
        37 => -1.0037 + 1.0665 * log_moves + 0.0381 * e,
        38 => -1.0599 + 1.0731 * log_moves + 0.0371 * e,
        39 => -1.1244 + 1.0808 * log_moves + 0.0359 * e,
        40 => -1.1983 + 1.0896 * log_moves + 0.0345 * e,
        41 => -1.2822 + 1.0993 * log_moves + 0.0330 * e,
        42 => -1.3771 + 1.1098 * log_moves + 0.0312 * e,
        43 => -1.4839 + 1.1212 * log_moves + 0.0293 * e,
        44 => -1.6032 + 1.1338 * log_moves + 0.0272 * e,
        45 => -1.7360 + 1.1475 * log_moves + 0.0249 * e,
        46 => -1.8831 + 1.1625 * log_moves + 0.0225 * e,
        47 => -2.0452 + 1.1790 * log_moves + 0.0200 * e,
        48 => -2.2231 + 1.1971 * log_moves + 0.0174 * e,
        49 => -2.4175 + 1.2170 * log_moves + 0.0147 * e,
        50 => -2.6290 + 1.2388 * log_moves + 0.0120 * e,
        51 => -2.8583 + 1.2627 * log_moves + 0.0092 * e,
        52 => -3.1060 + 1.2890 * log_moves + 0.0064 * e,
        53 => -3.3724 + 1.3177 * log_moves + 0.0036 * e,
        54 => -3.6580 + 1.3492 * log_moves + 0.0008 * e,
        55 => -3.9628 + 1.3837 * log_moves - 0.0020 * e,
        56 => -4.2865 + 1.4215 * log_moves - 0.0047 * e,
        57 => -4.6287 + 1.4627 * log_moves - 0.0074 * e,
        58 => -4.9884 + 1.5077 * log_moves - 0.0100 * e,
        59 => -5.3642 + 1.5568 * log_moves - 0.0124 * e,
        60 => -5.7541 + 1.6102 * log_moves - 0.0146 * e,
        61 => -6.1558 + 1.6681 * log_moves - 0.0166 * e,
        62 => -6.5667 + 1.7308 * log_moves - 0.0183 * e,
        63 => -6.9835 + 1.7985 * log_moves - 0.0196 * e,
        _ => {
            // Should be unreachable (empties is 0..63).
            debug_assert!(false, "n_empties out of range: {n_empties}");
            0.0
        }
    }
}

#[inline]
fn log_disproof_number(n_empties: u8, moves_player: u8, error: i32) -> f64 {
    let log_moves = log_moves(moves_player);
    let e = clamp_error(error);

    match n_empties {
        0 => 0.0290 + 0.9975 * log_moves - 0.0171 * e,
        1 => -0.1664 + 1.0233 * log_moves - 0.0278 * e,
        2 => -0.2940 + 1.0284 * log_moves - 0.0353 * e,
        3 => -0.3280 + 1.0000 * log_moves - 0.0379 * e,
        4 => -0.2830 + 0.9597 * log_moves - 0.0399 * e,
        5 => -0.2375 + 0.9234 * log_moves - 0.0406 * e,
        6 => -0.2218 + 0.9086 * log_moves - 0.0410 * e,
        7 => -0.2353 + 0.9107 * log_moves - 0.0414 * e,
        8 => -0.2632 + 0.9170 * log_moves - 0.0415 * e,
        9 => -0.2912 + 0.9220 * log_moves - 0.0414 * e,
        10 => -0.3155 + 0.9265 * log_moves - 0.0412 * e,
        11 => -0.3350 + 0.9305 * log_moves - 0.0411 * e,
        12 => -0.3510 + 0.9342 * log_moves - 0.0411 * e,
        13 => -0.3649 + 0.9375 * log_moves - 0.0410 * e,
        14 => -0.3785 + 0.9409 * log_moves - 0.0410 * e,
        15 => -0.3922 + 0.9447 * log_moves - 0.0409 * e,
        16 => -0.4056 + 0.9483 * log_moves - 0.0409 * e,
        17 => -0.4178 + 0.9516 * log_moves - 0.0408 * e,
        18 => -0.4287 + 0.9546 * log_moves - 0.0407 * e,
        19 => -0.4392 + 0.9575 * log_moves - 0.0406 * e,
        20 => -0.4500 + 0.9606 * log_moves - 0.0404 * e,
        21 => -0.4613 + 0.9640 * log_moves - 0.0403 * e,
        22 => -0.4732 + 0.9676 * log_moves - 0.0400 * e,
        23 => -0.4859 + 0.9714 * log_moves - 0.0397 * e,
        24 => -0.4998 + 0.9756 * log_moves - 0.0394 * e,
        25 => -0.5154 + 0.9803 * log_moves - 0.0389 * e,
        26 => -0.5329 + 0.9855 * log_moves - 0.0384 * e,
        27 => -0.5531 + 0.9914 * log_moves - 0.0378 * e,
        28 => -0.5762 + 0.9981 * log_moves - 0.0370 * e,
        29 => -0.6029 + 1.0058 * log_moves - 0.0361 * e,
        30 => -0.6336 + 1.0146 * log_moves - 0.0352 * e,
        31 => -0.6687 + 1.0248 * log_moves - 0.0340 * e,
        32 => -0.7087 + 1.0363 * log_moves - 0.0327 * e,
        33 => -0.7541 + 1.0494 * log_moves - 0.0312 * e,
        34 => -0.8055 + 1.0642 * log_moves - 0.0295 * e,
        35 => -0.8635 + 1.0809 * log_moves - 0.0277 * e,
        36 => -0.9289 + 1.0998 * log_moves - 0.0256 * e,
        37 => -1.0023 + 1.1212 * log_moves - 0.0233 * e,
        38 => -1.0846 + 1.1452 * log_moves - 0.0207 * e,
        39 => -1.1764 + 1.1723 * log_moves - 0.0179 * e,
        40 => -1.2784 + 1.2025 * log_moves - 0.0148 * e,
        41 => -1.3914 + 1.2364 * log_moves - 0.0115 * e,
        42 => -1.5161 + 1.2740 * log_moves - 0.0080 * e,
        43 => -1.6533 + 1.3160 * log_moves - 0.0043 * e,
        44 => -1.8038 + 1.3625 * log_moves - 0.0004 * e,
        45 => -1.9683 + 1.4141 * log_moves + 0.0037 * e,
        46 => -2.1476 + 1.4710 * log_moves + 0.0081 * e,
        47 => -2.3424 + 1.5338 * log_moves + 0.0127 * e,
        48 => -2.5534 + 1.6031 * log_moves + 0.0174 * e,
        49 => -2.7812 + 1.6793 * log_moves + 0.0223 * e,
        50 => -3.0263 + 1.7630 * log_moves + 0.0273 * e,
        51 => -3.2892 + 1.8546 * log_moves + 0.0324 * e,
        52 => -3.5704 + 1.9542 * log_moves + 0.0374 * e,
        53 => -3.8699 + 2.0621 * log_moves + 0.0423 * e,
        54 => -4.1878 + 2.1782 * log_moves + 0.0470 * e,
        55 => -4.5236 + 2.3022 * log_moves + 0.0515 * e,
        56 => -4.8763 + 2.4338 * log_moves + 0.0558 * e,
        57 => -5.2440 + 2.5719 * log_moves + 0.0597 * e,
        58 => -5.6240 + 2.7147 * log_moves + 0.0631 * e,
        59 => -6.0125 + 2.8594 * log_moves + 0.0661 * e,
        60 => -6.4048 + 3.0029 * log_moves + 0.0685 * e,
        61 => -6.7953 + 3.1411 * log_moves + 0.0702 * e,
        62 => -7.1769 + 3.2693 * log_moves + 0.0714 * e,
        63 => -7.5411 + 3.3820 * log_moves + 0.0718 * e,
        _ => {
            debug_assert!(false, "n_empties out of range: {n_empties}");
            0.0
        }
    }
}

// --- Precomputed tables ---

struct ProofDisproofNumberData {
    #[allow(dead_code)]
    proof_number: Box<[u8]>,
    #[allow(dead_code)]
    disproof_number: Box<[u8]>,
    disproof_number_over_prob: Box<[i32]>,
    #[allow(dead_code)]
    byte_to_proof_number: [f64; 256],
}

impl ProofDisproofNumberData {
    fn new() -> Self {
        let mut proof_number = vec![0u8; PROOF_TABLE_LEN];
        let mut disproof_number = vec![0u8; PROOF_TABLE_LEN];
        let mut disproof_number_over_prob = vec![0i32; PROOF_TABLE_LEN];

        // byte -> proof number table
        let mut byte_to_proof_number = [0.0f64; 256];
        for b in 0u16..=255 {
            byte_to_proof_number[b as usize] = byte_to_proof_number_explicit(b as u8);
        }

        let max_i32 = (i32::MAX - 2) as f64;

        for empties in 0u8..64 {
            for moves in 0u8..=15 {
                for delta in -128i32..=128 {
                    let off = data_to_proof_number_offset(empties, moves, delta);

                    let pn = bound(log_proof_number(empties, moves, delta).exp());
                    let dn = bound(log_disproof_number(empties, moves, delta).exp());

                    proof_number[off] = proof_number_to_byte(pn);
                    disproof_number[off] = proof_number_to_byte(dn);

                    let prob = probability_explicit(1, empties, delta);
                    let mut over = if prob > 0.0 { dn / prob } else { max_i32 };
                    if over > max_i32 {
                        over = max_i32;
                    }
                    disproof_number_over_prob[off] = over.round() as i32;
                }
            }
        }

        Self {
            proof_number: proof_number.into_boxed_slice(),
            disproof_number: disproof_number.into_boxed_slice(),
            disproof_number_over_prob: disproof_number_over_prob.into_boxed_slice(),
            byte_to_proof_number,
        }
    }
}

static PROOF_DISPROOF_DATA: OnceLock<ProofDisproofNumberData> = OnceLock::new();

#[inline(always)]
fn data() -> &'static ProofDisproofNumberData {
    PROOF_DISPROOF_DATA.get_or_init(ProofDisproofNumberData::new)
}

// --- Public API ---

/// Sensei-style `DisproofNumberOverProb`.
///
/// - `player_bits` / `opponent_bits` are bitboards for the side to move.
/// - `lower_eval_large` and `approx_eval_large` are in `EvalLarge` units (1/8 disc).
///
/// This matches Sensei's:
/// ```cpp
/// DisproofNumberOverProb(player, opponent, lower, approx_eval)
///   = table[empties, n_moves(player), (approx_eval - lower) >> 3]
/// ```
#[inline(always)]
pub fn disproof_number_over_prob(
    player_bits: u64,
    opponent_bits: u64,
    lower_eval_large: i32,
    approx_eval_large: i32,
) -> i32 {
    let occupied = player_bits | opponent_bits;
    let empties_raw = (64u32 - occupied.count_ones()) as u8;
    let empties = empties_raw.min(63);

    // Sensei caps move count at 15 for table indexing.
    let n_moves = legal_moves(player_bits, opponent_bits).count_ones().min(15) as u8;

    // Quantize EvalLarge (1/8 disc) -> discs.
    let eval_delta_raw = (approx_eval_large - lower_eval_large) >> 3;
    let eval_delta = eval_delta_raw.clamp(-128, 128);

    let off = data_to_proof_number_offset(empties, n_moves, eval_delta);
    data().disproof_number_over_prob[off]
}

/// Convert a compressed proof-number byte back into its approximate floating value.
#[inline(always)]
#[allow(dead_code)]
pub fn byte_to_proof_number(byte: u8) -> f64 {
    data().byte_to_proof_number[byte as usize]
}
