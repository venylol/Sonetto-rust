//! sonetto-wasm: wasm-bindgen wrapper for `sonetto-core`.
//!
//! # JS 使用方式（强烈建议）
//!
//! wasm-pack 生成的 JS **默认导出**通常也叫 `init`（用于实例化 wasm）。本 crate 同时导出一个
//! **命名导出** `init()`（用于安装 `console_error_panic_hook`），因此你需要在 JS 里重命名其一：
//!
//! ```js
//! import initWasm, {
//!   init as initPanicHook,
//!   engine_new,
//!   engine_set_weights_egev2,
//!   engine_best_move,
//! } from "sonetto-wasm";
//!
//! await initWasm();
//! initPanicHook();
//! ```
//!
//! # 线程 / Rayon（可选）
//!
//! 如果启用了 `wasm-bindgen-rayon`（并在构建时开启 wasm threads/atomics），本模块会 re-export
//! `init_thread_pool`（在 JS 侧名字是 `initThreadPool`）。典型用法：
//!
//! ```js
//! import initWasm, { initThreadPool } from "sonetto-wasm";
//! await initWasm();
//! await initThreadPool(n);
//! ```
//!
//! ## 部署必读：SharedArrayBuffer 需要 COOP/COEP
//! 启用 wasm 线程（`SharedArrayBuffer`）时，你的页面必须带上：
//! - `Cross-Origin-Opener-Policy: same-origin`
//! - `Cross-Origin-Embedder-Policy: require-corp`
//!
//! 否则浏览器会禁用 `SharedArrayBuffer`，线程池初始化会失败。
//!
//! # 坐标系统（非常重要）
//!
//! JS 侧传入的 `board_arr`（Uint8Array(64)）的索引 `0..63` 被视为 **ext packed**：
//! `ext = (col<<3)|row`（UI/协议用），必须通过 `sonetto_core::coord::ext_to_bitpos` 转成
//! 内部 bitboard 的 `bitpos = (row<<3)|col` 才能喂给引擎。
//!
//! 本文件在边界处完成该转换，避免内部模块坐标混用。

use wasm_bindgen::prelude::*;

use sonetto_core::{
    backend::BackendKind,
    board::{Board, Color, Undo},
    coord::{bitpos_move_to_ext_move, ext_to_bitpos, Move, PASS},
    egev2::{decode_egev2, encode_egev2, EVAL_MAX},
    eval::{build_sonetto_feature_defs_and_occ, score_disc, warm_up_eval_tables, Weights, N_PATTERN_FEATURES},
    features::{
        swap::SwapTables,
        update::recompute_features_in_place,
    },
    movegen::{legal_moves, push_moves_from_mask},
    search::{AnalyzeMode, AnalyzeTopNParams, AnalyzeTopNRequest, AnalyzeTopNStrategy, SearchLimits, Searcher, StopPolicy},
    sensei_ab::SenseiAlphaBeta,
    score::{Score, SCALE},
};

use libm::tanhf;

/// 安装 panic hook，让 Rust panic 在浏览器 console 里可读（带 stack trace）。
#[wasm_bindgen]
pub fn init() {
    // 重复调用是安全的（set_once）。
    console_error_panic_hook::set_once();
}

/// 如果启用了 `wasm-bindgen-rayon`，则 re-export 线程池初始化函数。
/// JS 侧名字是 `initThreadPool`。
#[cfg(feature = "wasm-bindgen-rayon")]
pub use wasm_bindgen_rayon::init_thread_pool;



#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeatureMode {
    /// No incremental features stored on the board.
    None,
    /// Legacy: `feat_id_abs` is interpreted as 64 per-square digits (0/1/2 for empty/black/white).
    SqAbsDigits,
    /// P0-3: `feat_id_abs` stores 64 absolute ternary pattern IDs for EGEV2's symmetry-expanded patterns.
    Egev2PatternIds,
}
#[wasm_bindgen]
pub struct Engine {
    searcher: Searcher,
    sensei_ab: SenseiAlphaBeta,
    backend_kind: BackendKind,
    last_backend_used: BackendKind,
    feat_len: usize,
    feature_mode: FeatureMode,

    // ---------------------------------------------------------------------
    // P0-2: Inference scratch
    // ---------------------------------------------------------------------
    //
    // Avoid per-call heap allocations at the WASM boundary (best_move / analyze).
    // We reuse this `Board` and only overwrite bitboards + bookkeeping fields.
    scratch_board: Board,

    // ---------------------------------------------------------------------
    // Training state (runs entirely inside the Rust/WASM engine)
    // ---------------------------------------------------------------------
    train_board: Board,
    train_rng: u64,
    train_gens: u64,
}


impl Engine {
    #[inline(always)]
    fn rng_next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.train_rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.train_rng = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    #[inline(always)]
    fn rng_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.rng_next_u64() as usize) % n
    }

    fn reset_training_board(&mut self) {
        self.train_board = Board::new_start(self.feat_len);
        recompute_features_in_place(&mut self.train_board, &self.searcher.occ);
    }

    fn advance_training_board_one_ply(&mut self) {
        let me = self.train_board.player;
        let opp = self.train_board.opponent;

        let mask = legal_moves(me, opp);
        if mask == 0 {
            let opp_mask = legal_moves(opp, me);
            if opp_mask == 0 {
                // Game over
                self.reset_training_board();
                return;
            }
            // Pass
            let mut u = Undo::default();
            let _ = self
                .train_board
                .apply_move_with_occ(PASS, &mut u, Some(&self.searcher.occ));
            return;
        }

        let mut moves = [PASS; 64];
        let n = push_moves_from_mask(mask, &mut moves);
        let mv = moves[self.rng_usize(n as usize)];
        let mut u = Undo::default();
        let _ = self
            .train_board
            .apply_move_with_occ(mv, &mut u, Some(&self.searcher.occ));
    }

    /// One training step on the current `train_board`, then advances the board by one ply.
    /// Returns the absolute error in "stone" units.
    fn train_one_step(&mut self, _lr: f32) -> f32 {
        // Forward-only: align with Egaroucid's mid_evaluate semantics.
        let sd = score_disc(&self.train_board, &self.searcher.weights);
        let y = tanhf((sd as f32) / 50.0);

        // Same target as phase1: current disc diff from side-to-move POV.
        let me_bits = self.train_board.player;
        let opp_bits = self.train_board.opponent;
        let disc_diff = (me_bits.count_ones() as i32) - (opp_bits.count_ones() as i32);
        let t = tanhf((disc_diff as f32) / 50.0);

        let abs_err = (y - t).abs();

        // Keep evolving the self-play position so UI behaviour stays similar.
        self.advance_training_board_one_ply();
        self.train_gens = self.train_gens.wrapping_add(1);

        abs_err
    }

    /// Supervised one-sample train step (dataset-driven).
    ///
    /// Returns per-sample MSE in `tanh(eval/50)` space.
    fn train_supervised_one(&mut self, board: &Board, target_stone: f32, _lr: f32) -> f32 {
        // Forward-only: no parameter updates in this phase.
        let sd = score_disc(board, &self.searcher.weights);
        let y = tanhf((sd as f32) / 50.0);
        let t = tanhf(target_stone / 50.0);
        let diff = y - t;
        diff * diff
    }

    /// Supervised loss only (no updates).
    /// Returns per-sample MSE in `tanh(eval/50)` space.
    fn loss_supervised_one(&self, board: &Board, target_stone: f32) -> f32 {
        let sd = score_disc(board, &self.searcher.weights);
        let y = tanhf((sd as f32) / 50.0);
        let t = tanhf(target_stone / 50.0);
        let diff = y - t;
        diff * diff
    }

    /// Keep the Sensei backend in sync with the current `Searcher` context.
    ///
    /// In this project the evaluation pipeline (EGEV2 weights + feature defs + occ map)
    /// must stay identical across backends. We rebuild the Sensei backend on demand
    /// when weights change.
    #[inline]
    fn sync_sensei_from_searcher(&mut self) {
        self.sensei_ab = SenseiAlphaBeta::new_with_hash_size_mb(
            self.searcher.tt_mb(),
            self.searcher.weights.clone(),
            self.searcher.feats.clone(),
            self.searcher.swap.clone(),
            self.searcher.occ.clone(),
        );
    }

    /// Fill the reusable `scratch_board` from an **ext-packed** 64-byte board.
    ///
    /// This is a hot WASM boundary helper: by reusing the same `Board` and its
    /// `feat_id_abs` buffer, we avoid allocator pressure and reduce UI stutter.
    #[inline(always)]
    fn fill_scratch_from_ext(&mut self, board_arr: &[u8], side: Color) {
        // Avoid trapping the Wasm instance on malformed inputs.
        // Missing squares are treated as empty; extra bytes are ignored.

        // board_arr(ext packed indices) -> absolute bitboards(internal bitpos)
        // `bits_by_color[Black]=black`, `bits_by_color[White]=white`.
        let mut bits_by_color = [0u64; 2];
        let n = board_arr.len().min(64);
        for ext_usize in 0..n {
            let ext = ext_usize as u8;
            let v = board_arr[ext_usize];
            if v == 0 {
                continue;
            }
            let bitpos = ext_to_bitpos(ext);
            let bit = 1u64 << (bitpos as u64);
            match v {
                1 => bits_by_color[Color::Black.idx()] |= bit,
                2 => bits_by_color[Color::White.idx()] |= bit,
                _ => {
                    // invalid value: treat as empty
                }
            }
        }

        let black = bits_by_color[Color::Black.idx()];
        let white = bits_by_color[Color::White.idx()];
        let (player_bits, opponent_bits) = if side == Color::Black {
            (black, white)
        } else {
            (white, black)
        };

        let occ = black | white;
        let empty_count = 64u8.saturating_sub(occ.count_ones() as u8);

        let b = &mut self.scratch_board;
        b.side = side;
        b.player = player_bits;
        b.opponent = opponent_bits;
        b.empty_count = empty_count;
        b.hash = sonetto_core::zobrist::compute_hash(bits_by_color, side);

        // Mark cache format as unknown/stale until the root recompute.
        b.feat_is_pattern_ids = false;
        if b.feat_id_abs.len() != self.feat_len {
            b.feat_id_abs.resize(self.feat_len, 0);
        } else {
            b.feat_id_abs.fill(0);
        }

        // Optional legacy per-square digits
        if self.feature_mode == FeatureMode::SqAbsDigits {
            // Ensure the buffer is at least 64.
            if b.feat_id_abs.len() < 64 {
                b.feat_id_abs.resize(64, 0);
            }
            for i in 0..64usize {
                let mask = 1u64 << (i as u64);
                b.feat_id_abs[i] = if (black & mask) != 0 {
                    Color::Black.digit_abs()
                } else if (white & mask) != 0 {
                    Color::White.digit_abs()
                } else {
                    0
                };
            }
        }
    }

}

#[inline(always)]
fn backend_kind_to_u8(k: BackendKind) -> u8 {
    match k {
        BackendKind::Sonetto => 0,
        BackendKind::SenseiAlphaBeta => 1,
        // `BackendKind` is `#[non_exhaustive]`.
        _ => 0,
    }
}

#[inline(always)]
fn backend_u8_to_kind(v: u8) -> BackendKind {
    match v {
        1 => BackendKind::SenseiAlphaBeta,
        _ => BackendKind::Sonetto,
    }
}

#[wasm_bindgen]
pub fn engine_new(hash_size_mb: usize) -> Engine {
    // P0-3: proactively initialize large OnceLock tables so the first search
    // doesn't "stall" the UI.
    warm_up_eval_tables();
    sonetto_core::stability::warm_up_stability_tables();

    // Also warm up the flip table if the experimental `flips_table` feature is enabled.
    #[cfg(feature = "flips_table")]
    sonetto_core::flips::warm_up_flip_tables();

    // Default engine configuration: start with zero weights. Host can call
    // `engine_set_weights_egev2` to load an Egaroucid `eval.egev2`.
    let weights = Weights::zeroed();

    // Build the Sonetto feature set + OccMap once (no heap allocations during search).
    let (feats, occ) = build_sonetto_feature_defs_and_occ();
    let swap = SwapTables::build_swap_tables();

    // P0-3: we maintain incremental absolute pattern IDs (64 features) during search.
    let feat_len = N_PATTERN_FEATURES;
    let feature_mode = FeatureMode::Egev2PatternIds;

    let searcher = Searcher::new(hash_size_mb, weights, feats, swap, occ);

    // Sensei-style alpha-beta backend (coexists with `Searcher`).
    // It uses the same evaluation pipeline, so it must be kept in sync when
    // weights change.
    let sensei_ab = SenseiAlphaBeta::new_with_hash_size_mb(
        hash_size_mb as usize,
        searcher.weights.clone(),
        searcher.feats.clone(),
        searcher.swap.clone(),
        searcher.occ.clone(),
    );

    // P0-2: scratch board reused by best-move / analyze entry points.
    let scratch_board = Board::new_empty(Color::Black, feat_len);

    // Training board starts from the initial position.
    let mut train_board = Board::new_start(feat_len);
    recompute_features_in_place(&mut train_board, &searcher.occ);

    Engine {
        searcher,
        sensei_ab,
        backend_kind: BackendKind::Sonetto,
        last_backend_used: BackendKind::Sonetto,
        feat_len,
        feature_mode,
        scratch_board,
        train_board,
        train_rng: 0x9e3779b97f4a7c15u64 ^ (hash_size_mb as u64).wrapping_mul(0xD1B54A32D192ED03),
        train_gens: 0,
    }
}

/// Select the search backend used by the additive `*_v3` entry points.
///
/// Values:
/// - `0` = Sonetto `Searcher` (default; keeps legacy behavior)
/// - `1` = Sensei-style alpha-beta backend
#[wasm_bindgen]
pub fn engine_set_backend(engine: &mut Engine, backend: u8) -> u8 {
    engine.backend_kind = backend_u8_to_kind(backend);
    backend_kind_to_u8(engine.backend_kind)
}

/// Current selected backend for the additive `*_v3` entry points.
#[wasm_bindgen]
pub fn engine_get_backend(engine: &Engine) -> u8 {
    backend_kind_to_u8(engine.backend_kind)
}

/// Nodes visited in the most recent search/analyze call.
///
/// Returned as `u32` for easy JS interop. If the internal counter exceeds
/// `u32::MAX`, the value is saturated.
#[wasm_bindgen]
pub fn engine_last_nodes(engine: &Engine) -> u32 {
    let n = engine.searcher.last_nodes();
    if n > (u32::MAX as u64) { u32::MAX } else { n as u32 }
}

/// Nodes visited in the most recent `*_v3` call.
///
/// Unlike [`engine_last_nodes`], this reports the count from whichever backend
/// was actually used by the last `best_move_v3/analyze_v3` invocation.
#[wasm_bindgen]
pub fn engine_last_nodes_v3(engine: &Engine) -> u32 {
    let n = match engine.last_backend_used {
        BackendKind::Sonetto => engine.searcher.last_nodes(),
        BackendKind::SenseiAlphaBeta => engine.sensei_ab.last_nodes(),
        _ => engine.searcher.last_nodes(),
    };
    if n > (u32::MAX as u64) { u32::MAX } else { n as u32 }
}


// -----------------------------------------------------------------------------
// Egaroucid eval.egev2 import/export
// -----------------------------------------------------------------------------

#[wasm_bindgen]
pub fn engine_set_weights_egev2(engine: &mut Engine, bytes: &[u8]) -> bool {
    let params = decode_egev2(bytes);
    if params.len() != Weights::expected_len() {
        return false;
    }

    // Install into the live evaluation weights.
    engine.searcher.weights = Weights::from_vec(params);

    // Any TT entries computed under the old weights are now stale.
    engine.searcher.tt.clear();

    // Keep Sensei backend consistent with the current eval pipeline.
    engine.sync_sensei_from_searcher();
    true
}

#[wasm_bindgen]
pub fn engine_get_weights_egev2(engine: &Engine) -> Vec<u8> {
    encode_egev2(engine.searcher.weights.as_slice())
}

// -----------------------------------------------------------------------------
// Legacy flat weights import/export (Float32Array)
// -----------------------------------------------------------------------------

/// Legacy weights setter.
///
/// Accepts a `Float32Array` (or JS array) and converts each value to `i16`
/// (rounded, clamped to Egaroucid's `EVAL_MAX`).
///
/// This exists purely for backward compatibility with older Sonetto ZIPs that
/// stored weights as raw little-endian `f32` blobs (`weights.f32`).
///
/// Returns `true` when the provided vector length matches the current
/// `eval.egev2` expected parameter count.
#[wasm_bindgen]
pub fn engine_set_weights_flat(engine: &mut Engine, weights_flat: &[f32]) -> bool {
    // Empty means "reset to zero" (missing params read as 0).
    if weights_flat.is_empty() {
        engine.searcher.weights = Weights::zeroed();
        engine.searcher.tt.clear();
        engine.sync_sensei_from_searcher();
        return true;
    }

    // Convert f32 -> i16 (Egaroucid semantics).
    let mut params: Vec<i16> = Vec::with_capacity(weights_flat.len());
    let lo = -(EVAL_MAX as i32);
    let hi = EVAL_MAX as i32;

    for &f in weights_flat {
        if !f.is_finite() {
            params.push(0);
            continue;
        }
        // Round-to-nearest, ties away from zero.
        let r = if f >= 0.0 { (f + 0.5).floor() } else { (f - 0.5).ceil() };
        let mut v = r as i32;
        if v < lo {
            v = lo;
        } else if v > hi {
            v = hi;
        }
        params.push(v as i16);
    }

    engine.searcher.weights = Weights::from_vec(params);
    // Any TT entries computed under the old weights are now stale.
    engine.searcher.tt.clear();

    // Keep Sensei backend consistent with the current eval pipeline.
    engine.sync_sensei_from_searcher();

    weights_flat.len() == Weights::expected_len()
}

/// Legacy weights getter.
///
/// Returns the current evaluation parameters as `Float32Array` (via wasm-bindgen
/// mapping for `Vec<f32>`).
#[wasm_bindgen]
pub fn engine_get_weights_flat(engine: &Engine) -> Vec<f32> {
    engine
        .searcher
        .weights
        .params
        .iter()
        .map(|&v| v as f32)
        .collect()
}

#[wasm_bindgen]
pub fn engine_set_training_gens(engine: &mut Engine, gens: f64) {
    if gens.is_finite() && gens >= 0.0 {
        engine.train_gens = gens.round() as u64;
    }
}

#[wasm_bindgen]
pub fn engine_get_training_gens(engine: &Engine) -> f64 {
    engine.train_gens as f64
}

/// Current training board as an **ext packed** 64-byte array.
///
/// Index is `(col<<3)|row`, value is 0/1/2 (empty/black/white).
#[wasm_bindgen]
pub fn engine_get_training_board(engine: &Engine) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let black = engine.train_board.bits_of(Color::Black);
    let white = engine.train_board.bits_of(Color::White);
    for bitpos in 0u8..=63 {
        let ext = sonetto_core::coord::bitpos_to_ext(bitpos);
        let mask = 1u64 << (bitpos as u64);
        out[ext as usize] = if (black & mask) != 0 {
            1
        } else if (white & mask) != 0 {
            2
        } else {
            0
        };
    }
    out
}

#[wasm_bindgen]
pub fn engine_get_training_player(engine: &Engine) -> u8 {
    match engine.train_board.side {
        Color::Black => 1,
        Color::White => 2,
    }
}

// -----------------------------------------------------------------------------
// Training entry point
// -----------------------------------------------------------------------------

/// Perform `batch_size` gradient steps.
///
/// Returns average MAE in "disc" units (same unit as UI `Stone Loss (MAE)`).
#[wasm_bindgen]
pub fn engine_train_batch(engine: &mut Engine, batch_size: u32, learning_rate: f32) -> f32 {
    if batch_size == 0 {
        return 0.0;
    }

    let mut sum_err: f32 = 0.0;
    for _ in 0..batch_size {
        sum_err += engine.train_one_step(learning_rate);
    }
    sum_err / (batch_size as f32)
}


// -----------------------------------------------------------------------------
// Supervised training / validation (dataset-driven)
// -----------------------------------------------------------------------------

/// Train on a supervised batch.
///
/// Inputs:
/// - `boards_flat`: concatenated `batch * 64` bytes in **ext-packed** order.
///   Each entry is 0 (empty), 1 (player / X), 2 (opponent / O).
/// - `targets`: length = batch; values are target evaluations in **stone units**
///   (estimated final stone difference from the player-to-move perspective).
///
/// Returns: average **MSE** over the batch, where the compared values are
/// `tanh(eval/50)` (so the loss is in a stable [-1,1] range). This is the
/// same space used for backprop.
#[wasm_bindgen]
pub fn engine_train_supervised_batch(
    engine: &mut Engine,
    boards_flat: &[u8],
    targets: &[f32],
    learning_rate: f32,
) -> f32 {
    let n = targets.len();
    if n == 0 {
        return 0.0;
    }
    if boards_flat.len() != n * 64 {
        // Avoid trapping the Wasm instance on malformed inputs.
        return f32::NAN;
    }

    let mut sum_loss: f32 = 0.0;
    for i in 0..n {
        let start = i * 64;
        let end = start + 64;
        let mut board = build_board_from_ext_arr(engine, &boards_flat[start..end], Color::Black);
        // Important: dataset boards arrive without feature ids.
        recompute_features_in_place(&mut board, &engine.searcher.occ);

        sum_loss += engine.train_supervised_one(&board, targets[i], learning_rate);
    }

    sum_loss / (n as f32)
}

/// Compute validation loss (no weight updates).
///
/// Same input format as [`engine_train_supervised_batch`].
/// Returns average MSE in `tanh(eval/50)` space.
#[wasm_bindgen]
pub fn engine_loss_supervised_batch(engine: &Engine, boards_flat: &[u8], targets: &[f32]) -> f32 {
    let n = targets.len();
    if n == 0 {
        return 0.0;
    }
    if boards_flat.len() != n * 64 {
        // Avoid trapping the Wasm instance on malformed inputs.
        return f32::NAN;
    }

    let mut sum_loss: f32 = 0.0;
    for i in 0..n {
        let start = i * 64;
        let end = start + 64;
        let mut board = build_board_from_ext_arr(engine, &boards_flat[start..end], Color::Black);
        recompute_features_in_place(&mut board, &engine.searcher.occ);

        sum_loss += engine.loss_supervised_one(&board, targets[i]);
    }

    sum_loss / (n as f32)
}


// -----------------------------------------------------------------------------
// Board helpers
// -----------------------------------------------------------------------------

#[inline(always)]
fn score_to_disc_int_rounded(score: Score) -> i32 {
    // `Score` uses SCALE=32 units per disc. Convert to integer "disc" units.
    // Round-to-nearest (ties away from zero).
    if score >= 0 {
        ((score + (SCALE / 2)) / SCALE) as i32
    } else {
        ((score - (SCALE / 2)) / SCALE) as i32
    }
}


#[inline(always)]
fn build_board_from_ext_arr(engine: &Engine, board_arr: &[u8], side: Color) -> Board {
    // Avoid trapping the Wasm instance on malformed inputs.
    // If the caller provides a slice that is not length-64, we interpret missing
    // squares as empty and ignore any extra bytes.

    // board_arr(ext packed indices) -> absolute bitboards(internal bitpos)
    // `bits_by_color[Black]=black`, `bits_by_color[White]=white`.
    let mut bits_by_color = [0u64; 2];
    let n = board_arr.len().min(64);
    for ext_usize in 0..n {
        let ext = ext_usize as u8;
        let v = board_arr[ext_usize];
        if v == 0 {
            continue;
        }
        let bitpos = ext_to_bitpos(ext);
        let bit = 1u64 << (bitpos as u64);
        match v {
            1 => bits_by_color[Color::Black.idx()] |= bit,
            2 => bits_by_color[Color::White.idx()] |= bit,
            _ => {
                // invalid value: treat as empty
            }
        }
    }

    let occ = bits_by_color[0] | bits_by_color[1];
    let empty_count = 64u8.saturating_sub(occ.count_ones() as u8);

    // P2-3: Board stores player/opponent bitboards; map from absolute color.
    let black = bits_by_color[Color::Black.idx()];
    let white = bits_by_color[Color::White.idx()];
    let (player_bits, opponent_bits) = if side == Color::Black { (black, white) } else { (white, black) };

    let mut board = Board {
        player: player_bits,
        opponent: opponent_bits,
        side,
        empty_count,
        hash: sonetto_core::zobrist::compute_hash(bits_by_color, side),
        feat_is_pattern_ids: false,
        feat_id_abs: vec![0u16; engine.feat_len],
    };

    // Optional legacy per-square digits
    if engine.feature_mode == FeatureMode::SqAbsDigits {
        // Defensive: ensure the per-square digit buffer is addressable.
        if board.feat_id_abs.len() < 64 {
            board.feat_id_abs.resize(64, 0u16);
        }
        for i in 0..64usize {
            let b = 1u64 << (i as u64);
            board.feat_id_abs[i] = if (bits_by_color[Color::Black.idx()] & b) != 0 {
                Color::Black.digit_abs()
            } else if (bits_by_color[Color::White.idx()] & b) != 0 {
                Color::White.digit_abs()
            } else {
                0
            };
        }
    }

    board
}

#[wasm_bindgen]
pub fn engine_best_move(engine: &mut Engine, board_arr: &[u8], player: u8, depth: u8) -> u8 {
    let side = match player {
        1 => Color::Black,
        2 => Color::White,
        _ => return 255u8,
    };

    // Fill the reusable scratch board (no heap allocations).
    engine.fill_scratch_from_ext(board_arr, side);
    let board = &mut engine.scratch_board;

    // 搜索
    const INF_I32: i32 = 1_000_000;
    let (/*score*/ _, mv_bit): (Score, Move) =
        engine.searcher.search(board, -INF_I32 as Score, INF_I32 as Score, depth);

    // 4) internal bitpos move -> ext packed move（PASS 保持 255）
    bitpos_move_to_ext_move(mv_bit)
}

/// Additive backend-aware best-move entry.
///
/// This keeps the legacy `engine_best_move` API intact while enabling runtime
/// backend switching via [`engine_set_backend`].
#[wasm_bindgen]
pub fn engine_best_move_v3(engine: &mut Engine, board_arr: &[u8], player: u8, depth: u8) -> u8 {
    match engine.backend_kind {
        BackendKind::Sonetto => {
            engine.last_backend_used = BackendKind::Sonetto;
            engine_best_move(engine, board_arr, player, depth)
        }
        BackendKind::SenseiAlphaBeta => {
            let side = match player {
                1 => Color::Black,
                2 => Color::White,
                _ => return 255u8,
            };

            engine.last_backend_used = BackendKind::SenseiAlphaBeta;
            engine.fill_scratch_from_ext(board_arr, side);
            let board = &mut engine.scratch_board;

            let (mv_bit, _score) = engine
                .sensei_ab
                .best_move(board, depth.max(1), SearchLimits::default());
            bitpos_move_to_ext_move(mv_bit)
        }
        _ => {
            engine.last_backend_used = BackendKind::Sonetto;
            engine_best_move(engine, board_arr, player, depth)
        }
    }
}

/// P0-4: Budgeted best-move helper.
///
/// This uses the Stage6 unified Top-N interface with:
/// - `top_n = 1`
/// - `strategy = Iterative`
/// - `stop_policy = GoodStop`
/// - `node_budget` override (0 = use core default)
///
/// It is designed for UI usage where a strict node budget is more predictable
/// than a fixed depth.
#[wasm_bindgen]
pub fn engine_best_move_v2(
    engine: &mut Engine,
    board_arr: &[u8],
    player: u8,
    mid_depth: u8,
    end_start: u8,
    node_budget: u32,
) -> u8 {
    let side = match player {
        1 => Color::Black,
        2 => Color::White,
        _ => return PASS,
    };

    engine.fill_scratch_from_ext(board_arr, side);
    let board = &mut engine.scratch_board;
    let empties = board.empty_count;

    // Match legacy auto-switch: only use exact solve in small endgames.
    let use_exact = empties <= end_start && empties <= 30;

    let mid_depth = mid_depth.max(1);
    let analyze_mode = if use_exact {
        AnalyzeMode::Exact
    } else {
        AnalyzeMode::Midgame { depth: mid_depth }
    };

    let mut params = AnalyzeTopNParams::default();
    if node_budget != 0 {
        params.node_budget = Some(node_budget as u64);
    }

    let mut req = AnalyzeTopNRequest::new(analyze_mode, 1);
    req.strategy = AnalyzeTopNStrategy::Iterative;
    req.stop_policy = StopPolicy::GoodStop;
    req.params = params;

    let pairs = engine.searcher.analyze_top_n(board, req).pairs;
    let mv_bit = pairs.first().map(|(m, _)| *m).unwrap_or(PASS);
    bitpos_move_to_ext_move(mv_bit)
}

#[wasm_bindgen]
pub fn engine_analyze(
    engine: &mut Engine,
    board_arr: &[u8],
    player: u8,
    mid_depth: u8,
    end_start: u8,
    top_n: u8,
) -> Vec<i32> {
    let side = match player {
        1 => Color::Black,
        2 => Color::White,
        _ => return Vec::new(),
    };

    engine.fill_scratch_from_ext(board_arr, side);
    let board = &mut engine.scratch_board;
    let empties = board.empty_count;

    // Keep the API stable and safe.
    let top_n = (top_n as usize).clamp(1, 32);

    // Exact solver is only safe/efficient for small remaining empties.
    // (The UI should keep end_start in a small range, but we guard anyway.)
    let use_exact = empties <= end_start && empties <= 30;

    let pairs = if use_exact {
        // Exact endgame (perfect play to completion).
        //
        // P2-4: if wasm threads are available (rayon + SharedArrayBuffer)
        // and the JS side initialized the Rayon pool, we can evaluate moves
        // in parallel inside the wasm module.
        #[cfg(feature = "wasm-bindgen-rayon")]
        {
            if rayon::current_num_threads() > 1 {
                engine.searcher.analyze_top_n_exact_parallel(board, top_n)
            } else {
                engine.searcher.analyze_top_n_exact(board, top_n)
            }
        }
        #[cfg(not(feature = "wasm-bindgen-rayon"))]
        {
            engine.searcher.analyze_top_n_exact(board, top_n)
        }
    } else {
        // Midgame numeric evaluation (depth-limited).
        #[cfg(feature = "wasm-bindgen-rayon")]
        {
            if rayon::current_num_threads() > 1 {
                engine.searcher.analyze_top_n_mid_parallel(board, mid_depth, top_n)
            } else {
                engine.searcher.analyze_top_n_mid(board, mid_depth, top_n)
            }
        }
        #[cfg(not(feature = "wasm-bindgen-rayon"))]
        {
            engine.searcher.analyze_top_n_mid(board, mid_depth, top_n)
        }
    };

    let mut out: Vec<i32> = Vec::with_capacity(pairs.len() * 2);
    for (mv_bit, score) in pairs {
        let mv_ext = bitpos_move_to_ext_move(mv_bit) as i32;

        // Convert Score (scaled) to "disc" integer.
        let disc_int = score_to_disc_int_rounded(score);

        out.push(mv_ext);
        out.push(disc_int);
    }

    out
}

/// Additive backend-aware analyze entry.
///
/// The output format matches [`engine_analyze`]: a flat `Vec<i32>` containing
/// `(move_ext, eval_disc)` pairs.
#[wasm_bindgen]
pub fn engine_analyze_v3(
    engine: &mut Engine,
    board_arr: &[u8],
    player: u8,
    mid_depth: u8,
    end_start: u8,
    top_n: u8,
) -> Vec<i32> {
    match engine.backend_kind {
        BackendKind::Sonetto => {
            engine.last_backend_used = BackendKind::Sonetto;
            engine_analyze(engine, board_arr, player, mid_depth, end_start, top_n)
        }
        BackendKind::SenseiAlphaBeta => {
            let side = match player {
                1 => Color::Black,
                2 => Color::White,
                _ => return Vec::new(),
            };

            engine.last_backend_used = BackendKind::SenseiAlphaBeta;
            engine.fill_scratch_from_ext(board_arr, side);
            let board = &mut engine.scratch_board;
            let empties = board.empty_count;

            let top_n = (top_n as usize).clamp(1, 32);
            let use_exact = empties <= end_start && empties <= 30;
            let mid_depth = mid_depth.max(1);
            let analyze_mode = if use_exact {
                AnalyzeMode::Exact
            } else {
                AnalyzeMode::Midgame { depth: mid_depth }
            };

            let req = AnalyzeTopNRequest::new(analyze_mode, top_n)
                .with_strategy(AnalyzeTopNStrategy::Fixed);
            // If wasm threads are enabled (rayon + SharedArrayBuffer) and the JS
            // side initialized the Rayon pool, use the parallel root-split path.
            let pairs = {
                #[cfg(feature = "wasm-bindgen-rayon")]
                {
                    if rayon::current_num_threads() > 1 {
                        engine.sensei_ab.analyze_top_n_parallel(board, req).pairs
                    } else {
                        engine.sensei_ab.analyze_top_n(board, req).pairs
                    }
                }
                #[cfg(not(feature = "wasm-bindgen-rayon"))]
                {
                    engine.sensei_ab.analyze_top_n(board, req).pairs
                }
            };

            let mut out: Vec<i32> = Vec::with_capacity(pairs.len() * 2);
            for (mv_bit, score) in pairs {
                let mv_ext = bitpos_move_to_ext_move(mv_bit) as i32;
                let disc_int = score_to_disc_int_rounded(score);
                out.push(mv_ext);
                out.push(disc_int);
            }
            out
        }
        _ => {
            engine.last_backend_used = BackendKind::Sonetto;
            engine_analyze(engine, board_arr, player, mid_depth, end_start, top_n)
        }
    }
}





/// Stage 6 unified Top-N analysis entry.
///
/// This is an **additive** API: the legacy [`engine_analyze`] entry remains
/// available (and keeps its exact signature / behavior) for backward compatibility.
///
/// # Mode selection
///
/// `mode` selects which analysis backend to use:
///
/// - `0` = **Auto + Fixed** (legacy behavior): midgame fixed-depth, switch to exact
///         solve when `empties <= end_start` and `empties <= 30`.
/// - `1` = **Auto + Iterative** (Stage4): same auto switch, but uses iterative Top-N.
/// - `2` = **Midgame + Fixed**
/// - `3` = **Exact + Fixed**
/// - `4` = **Midgame + Iterative**
/// - `5` = **Exact + Iterative**
/// - `6` = **Derivative** (Stage5, Sensei `EvaluatorDerivative` port)
///
/// # Tuning knobs (defaults)
///
/// For the following parameters, pass `0` to use the core defaults:
///
/// - `seed_depth`: derivative seeding depth. `0` means "auto/adaptive" (Sensei-style).
/// - `aspiration_width_disc`: initial aspiration half-window in **disc units** (1 disc = 32 score).
/// - `node_budget`: node budget for budgeted modes (Iterative / Derivative).
/// - `tree_node_cap`: derivative arena capacity cap.
///
/// Sensei mapping (mechanism names):
/// - Fixed/Iterative: `EvaluatorAlphaBeta` (with aspiration windows for root searches).
/// - Iterative: Stage4 Top-N refinement loop.
/// - Derivative: `EvaluatorDerivative` scheduler (`AddChildren` / `SolvePosition`).
#[wasm_bindgen]
pub fn engine_analyze_v2(
    engine: &mut Engine,
    board_arr: &[u8],
    player: u8,
    mode: u8,
    mid_depth: u8,
    end_start: u8,
    top_n: u8,
    seed_depth: u8,
    aspiration_width_disc: i32,
    node_budget: u32,
    tree_node_cap: u32,
) -> Vec<i32> {
    let side = match player {
        1 => Color::Black,
        2 => Color::White,
        _ => return Vec::new(),
    };

    engine.fill_scratch_from_ext(board_arr, side);
    let board = &mut engine.scratch_board;
    let empties = board.empty_count;

    let top_n = (top_n as usize).clamp(1, 32);

    // Same safety guard as the legacy API.
    let use_exact = empties <= end_start && empties <= 30;

    let mid_depth = mid_depth.max(1);
    let mid_mode = AnalyzeMode::Midgame { depth: mid_depth };
    let exact_mode = AnalyzeMode::Exact;

    let (strategy, analyze_mode) = match mode {
        0 => (AnalyzeTopNStrategy::Fixed, if use_exact { exact_mode } else { mid_mode }),
        1 => (
            AnalyzeTopNStrategy::Iterative,
            if use_exact { exact_mode } else { mid_mode },
        ),
        2 => (AnalyzeTopNStrategy::Fixed, mid_mode),
        3 => (AnalyzeTopNStrategy::Fixed, exact_mode),
        4 => (AnalyzeTopNStrategy::Iterative, mid_mode),
        5 => (AnalyzeTopNStrategy::Iterative, exact_mode),
        6 => (
            AnalyzeTopNStrategy::Derivative,
            // Derivative ignores `AnalyzeMode` internally, but we keep it meaningful.
            if use_exact { exact_mode } else { mid_mode },
        ),
        _ => return Vec::new(),
    };

    // Build request using the core defaults, then apply user overrides.
    let mut params = AnalyzeTopNParams::default();
    if seed_depth != 0 {
        params.seed_depth = seed_depth;
    }
    if aspiration_width_disc > 0 {
        params.aspiration_width = (aspiration_width_disc as Score) * SCALE;
    }
    if node_budget != 0 {
        params.node_budget = Some(node_budget as u64);
    }
    if tree_node_cap != 0 {
        params.tree_node_cap = tree_node_cap as usize;
    }

    let mut req = AnalyzeTopNRequest::new(analyze_mode, top_n);
    req.strategy = strategy;
    req.params = params;

    // For iterative analysis we default to GoodStop (Stage4 heuristic early-stop).
    if strategy == AnalyzeTopNStrategy::Iterative {
        req.stop_policy = StopPolicy::GoodStop;
    }

    // P1-2: Fixed-depth analysis can exploit wasm threads (Rayon) via the
    // dedicated parallel helpers. Iterative/Derivative stay single-threaded.
    //
    // Note: when the user overrides `aspiration_width_disc`, we keep the canonical
    // Stage6 path to preserve tuning semantics.
    // The iterative/derivative Rayon helpers are optional and not used in the
    // default UI flow. Keep the symbols defined so the code compiles cleanly
    // when the `wasm-bindgen-rayon` feature is enabled.
    #[cfg(feature = "wasm-bindgen-rayon")]
    let want_parallel = false;
    #[cfg(feature = "wasm-bindgen-rayon")]
    let stop_policy = req.stop_policy;

    let pairs: Vec<(Move, Score)> = match (strategy, analyze_mode) {
        (AnalyzeTopNStrategy::Fixed, AnalyzeMode::Midgame { depth }) if aspiration_width_disc <= 0 => {
            #[cfg(feature = "wasm-bindgen-rayon")]
            {
                if rayon::current_num_threads() > 1 {
                    engine.searcher.analyze_top_n_mid_parallel(board, depth, top_n)
                } else {
                    engine.searcher.analyze_top_n_mid(board, depth, top_n)
                }
            }
            #[cfg(not(feature = "wasm-bindgen-rayon"))]
            {
                engine.searcher.analyze_top_n_mid(board, depth, top_n)
            }
        }
        (AnalyzeTopNStrategy::Fixed, AnalyzeMode::Exact) if aspiration_width_disc <= 0 => {
            #[cfg(feature = "wasm-bindgen-rayon")]
            {
                if rayon::current_num_threads() > 1 {
                    engine.searcher.analyze_top_n_exact_parallel(board, top_n)
                } else {
                    engine.searcher.analyze_top_n_exact(board, top_n)
                }
            }
            #[cfg(not(feature = "wasm-bindgen-rayon"))]
            {
                engine.searcher.analyze_top_n_exact(board, top_n)
            }
        }
        (AnalyzeTopNStrategy::Iterative, _) => {
            #[cfg(feature = "wasm-bindgen-rayon")]
            {
                if want_parallel {
                    let limits = sonetto_core::search::SearchLimits {
                        max_nodes: params.node_budget,
                    };
                    engine
                        .searcher
                        .analyze_top_n_iter_parallel(
                            board,
                            analyze_mode,
                            top_n,
                            limits,
                            stop_policy,
                            params.aspiration_width,
                        )
                        .pairs
                } else {
                    engine.searcher.analyze_top_n(board, req).pairs
                }
            }
            #[cfg(not(feature = "wasm-bindgen-rayon"))]
            {
                engine.searcher.analyze_top_n(board, req).pairs
            }
        }
        (AnalyzeTopNStrategy::Derivative, _) => {
            #[cfg(feature = "wasm-bindgen-rayon")]
            {
                if want_parallel {
                    let limits = sonetto_core::search::SearchLimits {
                        max_nodes: params.node_budget,
                    };
                    engine.searcher.analyze_top_n_derivative_parallel(
                        board,
                        top_n,
                        limits,
                        params.tree_node_cap,
                        params.seed_depth,
                    )
                } else {
                    engine.searcher.analyze_top_n(board, req).pairs
                }
            }
            #[cfg(not(feature = "wasm-bindgen-rayon"))]
            {
                engine.searcher.analyze_top_n(board, req).pairs
            }
        }
        _ => engine.searcher.analyze_top_n(board, req).pairs,
    };

    let mut out: Vec<i32> = Vec::with_capacity(pairs.len() * 2);
    for (mv_bit, score) in pairs {
        let mv_ext = bitpos_move_to_ext_move(mv_bit) as i32;
        let disc_int = score_to_disc_int_rounded(score);
        out.push(mv_ext);
        out.push(disc_int);
    }
    out
}
