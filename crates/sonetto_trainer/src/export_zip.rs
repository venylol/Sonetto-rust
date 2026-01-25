use serde_json::json;
use std::fs::{create_dir_all, File};
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipWriter};

use crate::datasets::training::{TrainCursor, TrainStats};
use crate::datasets::transcript::ValStats;

/// Export a Sonetto-compatible weights ZIP.
///
/// Minimum contents (per requirement):
/// - `eval.egev2`
/// - `manifest.json`
/// - `logs/trainer.log`
///
/// This mirrors the browser export shape, but does not attempt to embed
/// runnable (wasm/html) artifacts.
pub fn export_weights_zip(
    out_zip: &Path,
    eval_egev2_bytes: &[u8],
    weight_count: usize,
    gen: u64,
    lr_used: f32,
    lr_next: f32,
    train_cursor: Option<&TrainCursor>,
    train_stats: Option<&TrainStats>,
    last_val: Option<&ValStats>,
    logs: &str,
) -> io::Result<()> {
    if let Some(parent) = out_zip.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let created_utc = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| format!("unix:{}", d.as_secs()))
        .unwrap_or_else(|_| "unix:0".to_string());

    let train_cursor_json = train_cursor.map(|c| {
        json!({
            "file": c.file,
            "line": c.line,
            "samples_seen": c.samples_seen,
        })
    });

    let train_stats_json = train_stats.map(|s| {
        json!({
            "samples": s.samples,
            "avg_loss_tanh": s.avg_loss(),
            "updated_params": s.updated_params,
            "skipped_lines": s.skipped_lines,
            "bad_lines": s.bad_lines,
        })
    });

    let last_val_json = last_val.map(|v| {
        json!({
            "games": v.games,
            "positions": v.positions,
            "avg_loss_tanh": v.avg_loss(),
            "skipped_lines": v.skipped_lines,
            "bad_lines": v.bad_lines,
            "incomplete_games": v.incomplete_games,
        })
    });

    let engine_version = format!("sonetto_trainer@{}", env!("CARGO_PKG_VERSION"));

    // Keep the existing manifest keys for backward compatibility.
    // (Older tooling expects `format`/`version`/`gen`/`param_count`/`files`.)
    //
    // At the same time, also emit the browser-style fields (`format_version`, `engine_version`,
    // `package_type`, `meta.generations`) so the exported ZIP can be imported back into the
    // web UI without losing the generation counter.
    let manifest = json!({
        // --- Legacy keys ---
        "format": "sonetto-weightzip",
        "version": 6,
        "gen": gen,
        "param_count": weight_count,
        "created_utc": created_utc,
        "files": [
            "eval.egev2",
            "manifest.json",
            "logs/trainer.log"
        ],

        // --- Browser-compatible keys (additive) ---
        "format_version": 2,
        "engine_version": engine_version,
        "package_type": "weights-only",
        "meta": {
            "generations": gen
        },

        // --- Shared / resume keys ---
        "weights": {
            "file": "eval.egev2",
            "dtype": "egev2-i16-rle",
            "count": weight_count
        },
        "training": {
            "lr_used": lr_used,
            "lr_next": lr_next,
            "cursor": train_cursor_json,
            "epoch_stats": train_stats_json
        },
        "validation": {
            "last": last_val_json
        }
    });

    let manifest_text = serde_json::to_string_pretty(&manifest)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let f = File::create(out_zip)?;
    let mut zip = ZipWriter::new(f);

    let opts = FileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .unix_permissions(0o644);

    zip.start_file("eval.egev2", opts)?;
    zip.write_all(eval_egev2_bytes)?;

    zip.start_file("manifest.json", opts)?;
    zip.write_all(manifest_text.as_bytes())?;

    zip.start_file("logs/trainer.log", opts)?;
    zip.write_all(logs.as_bytes())?;

    zip.finish()?;
    Ok(())
}
