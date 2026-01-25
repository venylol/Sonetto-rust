use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use sonetto_core::board::{Board, Color};
use sonetto_core::egev2::EVAL_MAX;
use sonetto_core::eval::Weights;
use sonetto_core::eval_egev2::{
    calc_active_features, calc_rev_idx, round_clamp_raw_to_disc, FEATURE_TO_PATTERN,
    PATTERN_OFFSETS, PATTERN_PARAMS_PER_PHASE, PARAMS_PER_PHASE, SCORE_MAX, STEP,
};

use crate::log::RunLog;
use crate::util::parse_board64;

/// n_appear values are clamped in Egaroucid to avoid vanishing LR.
///
/// See: `eval_optimizer_cuda.cu` where `host_n_appear_arr[i] = min(50, host_n_appear_arr[i])`.
pub const N_APPEAR_CAP: u16 = 50;

/// A coarse training cursor that can be recorded into `manifest.json`.
///
/// Note: phase6 trainer still scans the dataset sequentially each epoch. This
/// cursor is mainly provided so an external resume tool can skip work if desired.
#[derive(Debug, Default, Clone)]
pub struct TrainCursor {
    pub file: String,
    pub line: u64,
    pub samples_seen: u64,
}

/// Aggregate statistics for one training epoch.
#[derive(Debug, Default, Clone)]
pub struct TrainStats {
    pub samples: u64,
    pub loss_sum: f64,
    pub skipped_lines: u64,
    pub bad_lines: u64,

    /// Number of **unique** parameter indices updated (after batch aggregation).
    pub updated_params: u64,

    /// Last processed sample location.
    pub cursor: Option<TrainCursor>,
}

impl TrainStats {
    pub fn avg_loss(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.loss_sum / (self.samples as f64)
        }
    }
}

/// Build `n_appear` counts by **streaming once** through the training dataset.
///
/// Alignment notes:
/// - Counts are accumulated for `(idx, rev_idx)` for each feature.
/// - Counts are capped to [`N_APPEAR_CAP`].
/// - No training samples are kept in memory.
pub fn build_n_appear(
    train_dir: &Path,
    params_len: usize,
    log: &mut RunLog,
    strict: bool,
) -> io::Result<Vec<u16>> {
    let files = list_txt_files(train_dir)?;
    log.line(format!(
        "n_appear: streaming {} file(s) under {}",
        files.len(),
        train_dir.display()
    ));

    let mut n_appear: Vec<u16> = vec![0u16; params_len];
    let mut samples: u64 = 0;
    let mut skipped: u64 = 0;
    let mut bad: u64 = 0;

    for path in files {
        let f = File::open(&path)?;
        let br = BufReader::new(f);
        for (lineno, line_res) in br.lines().enumerate() {
            let line = line_res?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                skipped += 1;
                continue;
            }

            let (board, _target) = match parse_training_line(line) {
                Ok(Some(v)) => v,
                Ok(None) => {
                    skipped += 1;
                    continue;
                }
                Err(e) => {
                    bad += 1;
                    if strict {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("{}:{}: {e}", path.display(), lineno + 1),
                        ));
                    } else {
                        log.line(format!(
                            "n_appear: skip bad line {}:{}: {e}",
                            path.display(),
                            lineno + 1
                        ));
                        continue;
                    }
                }
            };

            let active = calc_active_features(&board);
            let phase_base = active.phase * PARAMS_PER_PHASE;

            // 64 symmetry features.
            for fi in 0..FEATURE_TO_PATTERN.len() {
                let pattern = FEATURE_TO_PATTERN[fi];
                let local_idx = active.idx_by_feature[fi];
                let base = phase_base + PATTERN_OFFSETS[pattern];

                let idx = base + local_idx as usize;
                bump_capped(&mut n_appear, idx);

                let rev_local = calc_rev_idx(pattern, local_idx);
                let rev_idx = base + rev_local as usize;
                bump_capped(&mut n_appear, rev_idx);
            }

            // eval_num feature: identity rev_idx (Egaroucid), but we still bump twice
            // to match the (idx, rev_idx) counting convention.
            let eval_num_idx = phase_base + PATTERN_PARAMS_PER_PHASE + active.num_player_discs as usize;
            bump_capped(&mut n_appear, eval_num_idx);
            bump_capped(&mut n_appear, eval_num_idx);

            samples += 1;
        }
    }

    log.line(format!(
        "n_appear: done, samples={}, skipped_lines={}, bad_lines={} (cap={})",
        samples, skipped, bad, N_APPEAR_CAP
    ));

    Ok(n_appear)
}

/// Run one training epoch with **incremental** parameter updates.
///
/// Key properties (phase6):
/// - Each sample touches only: feature `idx` + `rev_idx` + `eval_num`.
/// - Learning rate is scaled per-parameter: `lr = alpha_stab / n_appear[idx]`.
/// - Weight clamp: `[-4091, 4091]`.
/// - Prediction rounding/clamp semantics align with Egaroucid:
///   raw -> (+/-16) -> /32 -> clamp [-64, 64].
///
/// Dataset line format:
/// `64chars_board` + ` ` + `score`
pub fn run_training_epoch(
    train_dir: &Path,
    weights: &mut Weights,
    n_appear: &[u16],
    alpha_stab: f32,
    batch_size: usize,
    log: &mut RunLog,
    strict: bool,
) -> io::Result<TrainStats> {
    let files = list_txt_files(train_dir)?;
    log.line(format!(
        "training: scanning {} file(s) under {} (batch_size={}, alpha_stab={})",
        files.len(),
        train_dir.display(),
        batch_size,
        alpha_stab
    ));

    let mut stats = TrainStats::default();
    let mut cursor = TrainCursor::default();

    // We aggregate gradients inside each batch as a sparse list:
    // (global_param_idx, residual_raw).
    let mut grads: Vec<(usize, f32)> = Vec::with_capacity(batch_size.max(1) * 130);
    let mut batch_n: usize = 0;

    for path in files {
        let f = File::open(&path)?;
        let br = BufReader::new(f);

        for (lineno, line_res) in br.lines().enumerate() {
            let line = line_res?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                stats.skipped_lines += 1;
                continue;
            }

            let (board, target_disc) = match parse_training_line(line) {
                Ok(Some(v)) => v,
                Ok(None) => {
                    stats.skipped_lines += 1;
                    continue;
                }
                Err(e) => {
                    stats.bad_lines += 1;
                    if strict {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("{}:{}: {e}", path.display(), lineno + 1),
                        ));
                    } else {
                        log.line(format!(
                            "training: skip bad line {}:{}: {e}",
                            path.display(),
                            lineno + 1
                        ));
                        continue;
                    }
                }
            };

            let active = calc_active_features(&board);

            // --- prediction (raw sum) ---
            let phase_base = active.phase * PARAMS_PER_PHASE;
            let pred_raw = raw_sum(weights, phase_base, &active);
            let pred_disc = round_clamp_raw_to_disc(pred_raw) as f32;

            // --- loss tracking (kept from phase5: tanh-space MSE) ---
            let y = libm::tanhf(pred_disc / 50.0);
            let t = libm::tanhf(target_disc / 50.0);
            let d = y - t;
            stats.loss_sum += (d * d) as f64;
            stats.samples += 1;

            // --- residual in raw STEP units (Egaroucid-style) ---
            let target_clamped = target_disc.clamp(-(SCORE_MAX as f32), SCORE_MAX as f32);
            let target_raw = target_clamped * (STEP as f32);
            let pred_raw_clamped = clamp_raw_score(pred_raw) as f32;
            let residual_raw = target_raw - pred_raw_clamped;

            // --- sparse gradient accumulation (idx + rev_idx + eval_num) ---
            accumulate_sample_grads(&active, phase_base, residual_raw, &mut grads);

            batch_n += 1;
            cursor.file = path.display().to_string();
            cursor.line = (lineno as u64) + 1;
            cursor.samples_seen = stats.samples;

            if batch_n >= batch_size.max(1) {
                apply_sparse_batch_update(weights, n_appear, alpha_stab, &mut grads, &mut stats);
                batch_n = 0;
            }
        }
    }

    if !grads.is_empty() {
        apply_sparse_batch_update(weights, n_appear, alpha_stab, &mut grads, &mut stats);
    }

    stats.cursor = Some(cursor);

    log.line(format!(
        "training: epoch done, samples={}, avg_loss={:.6}, updated_params={}",
        stats.samples,
        stats.avg_loss(),
        stats.updated_params
    ));

    Ok(stats)
}

#[inline(always)]
fn bump_capped(arr: &mut [u16], idx: usize) {
    if let Some(x) = arr.get_mut(idx) {
        if *x < N_APPEAR_CAP {
            *x += 1;
        }
    }
}

#[inline(always)]
fn clamp_raw_score(raw: i32) -> i32 {
    let max_raw = SCORE_MAX * STEP;
    raw.clamp(-max_raw, max_raw)
}

#[inline(always)]
fn raw_sum(weights: &Weights, phase_base: usize, active: &sonetto_core::eval_egev2::ActiveFeatures) -> i32 {
    let mut raw: i32 = 0;

    // 64 symmetry features.
    for fi in 0..FEATURE_TO_PATTERN.len() {
        let pattern = FEATURE_TO_PATTERN[fi];
        let local = active.idx_by_feature[fi] as usize;
        let idx = phase_base + PATTERN_OFFSETS[pattern] + local;
        raw += *weights.params.get(idx).unwrap_or(&0) as i32;
    }

    // eval_num.
    let eval_num_idx = phase_base + PATTERN_PARAMS_PER_PHASE + active.num_player_discs as usize;
    raw += *weights.params.get(eval_num_idx).unwrap_or(&0) as i32;

    raw
}

#[inline(always)]
fn accumulate_sample_grads(
    active: &sonetto_core::eval_egev2::ActiveFeatures,
    phase_base: usize,
    residual_raw: f32,
    out: &mut Vec<(usize, f32)>,
) {
    // 64 symmetry features: (idx, rev_idx)
    for fi in 0..FEATURE_TO_PATTERN.len() {
        let pattern = FEATURE_TO_PATTERN[fi];
        let local_idx = active.idx_by_feature[fi];
        let base = phase_base + PATTERN_OFFSETS[pattern];

        let idx = base + local_idx as usize;
        out.push((idx, residual_raw));

        let rev_local = calc_rev_idx(pattern, local_idx);
        let rev_idx = base + rev_local as usize;
        out.push((rev_idx, residual_raw));
    }

    // eval_num: identity rev_idx, but still push twice to match Egaroucid's (idx, rev_idx) update.
    let eval_num_idx = phase_base + PATTERN_PARAMS_PER_PHASE + active.num_player_discs as usize;
    out.push((eval_num_idx, residual_raw));
    out.push((eval_num_idx, residual_raw));
}

fn apply_sparse_batch_update(
    weights: &mut Weights,
    n_appear: &[u16],
    alpha_stab: f32,
    grads: &mut Vec<(usize, f32)>,
    stats: &mut TrainStats,
) {
    if grads.is_empty() {
        return;
    }

    grads.sort_unstable_by_key(|(idx, _)| *idx);

    // Ensure we have unique access to the parameter buffer once,
    // then update it in-place.
    let params = weights.make_mut();

    let mut i = 0;
    while i < grads.len() {
        let idx = grads[i].0;
        let mut sum: f32 = 0.0;
        while i < grads.len() && grads[i].0 == idx {
            sum += grads[i].1;
            i += 1;
        }

        // lr = alpha_stab / n_appear[idx] (with min=1)
        let appear = n_appear.get(idx).copied().unwrap_or(0);
        let denom = if appear == 0 { 1.0 } else { appear as f32 };
        let lr = alpha_stab / denom;

        // grad = 2 * sum(residual)
        let delta = lr * 2.0 * sum;

        if let Some(w) = params.get_mut(idx) {
            let next = (*w as f32 + delta).round();
            let clamped = next.clamp(-(EVAL_MAX as f32), EVAL_MAX as f32) as i16;
            *w = clamped;
        }

        stats.updated_params += 1;
    }

    grads.clear();
}

fn parse_training_line(line: &str) -> Result<Option<(Board, f32)>, String> {
    let mut it = line.split_whitespace();
    let board_tok = match it.next() {
        Some(x) => x,
        None => return Ok(None),
    };
    let score_tok = match it.next() {
        Some(x) => x,
        None => return Err("missing score token".to_string()),
    };

    // Training dataset format is strictly: `64chars_board` + ` ` + `score`.
    // (Side-to-move is not encoded here; we default to Black.)
    let board = parse_board64(board_tok, Color::Black)?;
    let score: f32 = score_tok
        .parse()
        .map_err(|_| format!("invalid score: '{score_tok}'"))?;

    Ok(Some((board, score)))
}

fn list_txt_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();

    let rd = match fs::read_dir(dir) {
        Ok(x) => x,
        Err(e) => {
            return Err(io::Error::new(
                e.kind(),
                format!("unable to read training dir {}: {e}", dir.display()),
            ))
        }
    };

    for ent in rd {
        let ent = ent?;
        let p = ent.path();
        if p.is_file() {
            if let Some(ext) = p.extension() {
                if ext == "txt" {
                    out.push(p);
                }
            }
        }
    }

    out.sort();
    Ok(out)
}
