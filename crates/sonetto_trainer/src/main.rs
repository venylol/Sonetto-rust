mod cli;
mod config;
mod datasets;
mod export_zip;
mod log;
mod teacher;
mod util;

use std::fs;
use std::io;
use std::path::Path;

use sonetto_core::egev2::decode_egev2;
use sonetto_core::egev2::encode_egev2;
use sonetto_core::eval::Weights;

use crate::export_zip::export_weights_zip;
use crate::log::RunLog;
use crate::teacher::TeacherClient;

fn main() {
    let cfg = match cli::parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    let mut log = RunLog::new(!cfg.quiet);
    log.raw("sonetto_trainer: starting");
    log.line(format!("config: eval={} train_dir={} val_dir={} out_zip={} epochs={} batch_size={} gen={} strict={} teacher_url={}",
        cfg.eval_egev2_path.display(),
        cfg.train_dir.display(),
        cfg.val_dir.display(),
        cfg.out_zip.display(),
        cfg.epochs,
        cfg.batch_size,
        cfg.gen,
        cfg.strict,
        cfg.teacher_url.as_deref().unwrap_or("<none>")
    ));

    // Load existing weights.
    let mut weights = match load_weights(&cfg.eval_egev2_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to load weights: {e}");
            return;
        }
    };

    if !weights.is_valid() {
        // We keep evaluation code tolerant of short params (missing reads as 0),
        // but the trainer needs a full-length buffer for stable indexing.
        let expected = Weights::expected_len();
        log.line(format!(
            "loaded eval.egev2: {} params (expected {}), resizing with zeros",
            weights.params.len(),
            expected
        ));
        weights.make_mut().resize(expected, 0);
    } else {
        log.line(format!("loaded eval.egev2: {} params", weights.params.len()));
    }

    let teacher = cfg.teacher_url.as_ref().map(|u| TeacherClient::new(u.clone()));

    // =============================
    // Schedule: initial_validation -> training -> post_epoch_validation -> extra_validation -> lr_decay
    // =============================

    log.raw("stage: initial_validation");
    let mut last_val = match datasets::transcript::run_validation_final_score(
        &cfg.val_dir,
        &weights,
        &mut log,
        cfg.strict,
    ) {
        Ok(s) => Some(s),
        Err(e) => {
            log.line(format!("initial_validation: skipped ({e})"));
            None
        }
    };

    log.raw("stage: n_appear");
    let n_appear = match datasets::training::build_n_appear(
        &cfg.train_dir,
        weights.params.len(),
        &mut log,
        cfg.strict,
    ) {
        Ok(v) => v,
        Err(e) => {
            log.line(format!("n_appear: skipped ({e})"));
            // Fallback: treat everything as appearing once.
            vec![1u16; weights.params.len()]
        }
    };

    let mut lr = cfg.lr;
    let mut last_lr_used = lr;
    let mut last_train_cursor: Option<datasets::training::TrainCursor> = None;
    let mut last_train_stats: Option<datasets::training::TrainStats> = None;

    for epoch in 0..cfg.epochs {
        log.raw(format!("epoch: {}/{}", epoch + 1, cfg.epochs));

        log.raw("stage: training");
        last_lr_used = lr;
        let train = match datasets::training::run_training_epoch(
            &cfg.train_dir,
            &mut weights,
            &n_appear,
            lr,
            cfg.batch_size,
            &mut log,
            cfg.strict,
        ) {
            Ok(s) => Some(s),
            Err(e) => {
                log.line(format!("training: skipped ({e})"));
                None
            }
        };

        if let Some(s) = train.as_ref() {
            last_train_cursor = s.cursor.clone();
            last_train_stats = Some(s.clone());
        }

        log.raw("stage: post_epoch_validation");
        last_val = match datasets::transcript::run_validation_final_score(
            &cfg.val_dir,
            &weights,
            &mut log,
            cfg.strict,
        ) {
            Ok(s) => Some(s),
            Err(e) => {
                log.line(format!("post_epoch_validation: skipped ({e})"));
                None
            }
        };

        log.raw("stage: extra_validation");
        if let Some(t) = teacher.as_ref() {
            let _val_teacher = match datasets::transcript::run_validation_teacher(
                &cfg.val_dir,
                &weights,
                t,
                cfg.teacher_batch_size,
                &mut log,
                cfg.strict,
            ) {
                Ok(s) => Some(s),
                Err(e) => {
                    log.line(format!("extra_validation (teacher): skipped ({e})"));
                    None
                }
            };
        } else {
            log.line("extra_validation: skipped (no --teacher-url)".to_string());
        }

        log.raw("stage: lr_decay");
        lr *= cfg.lr_decay;
        log.line(format!("lr_decay: lr -> {lr}"));
    }

    // Export a weights-only ZIP.
    log.raw("stage: export");
    log.line(format!("export: writing {}", cfg.out_zip.display()));

    let eval_bytes = encode_egev2(weights.as_slice());

    let logs_text = log.into_string();
    if let Err(e) = export_weights_zip(
        &cfg.out_zip,
        &eval_bytes,
        weights.params.len(),
        cfg.gen,
        last_lr_used,
        lr,
        last_train_cursor.as_ref(),
        last_train_stats.as_ref(),
        last_val.as_ref(),
        &logs_text,
    ) {
        eprintln!("export failed: {e}");
        return;
    }

    if !cfg.quiet {
        println!("export ok: {}", cfg.out_zip.display());
    }
}

fn load_weights(path: &Path) -> io::Result<Weights> {
    let bytes = fs::read(path)?;

    // `sonetto_core::egev2::decode_egev2` is intentionally tolerant and returns an empty vec on
    // malformed input. The trainer, however, should fail fast on obviously broken inputs so we
    // don't silently train/export garbage.
    let params = decode_egev2(&bytes);
    if params.is_empty() {
        // Treat `n=0` as a valid (but useless) file; everything else is likely invalid.
        if bytes.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "eval.egev2 too small (missing header)",
            ));
        }
        let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap_or([0, 0, 0, 0]));
        if n != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "failed to decode eval.egev2 (empty output)",
            ));
        }
    }

    Ok(Weights::from_vec(params))
}
