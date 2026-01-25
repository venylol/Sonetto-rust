use std::path::PathBuf;

use crate::config::TrainerConfig;

/// Parse CLI args into [`TrainerConfig`].
///
/// Supported styles:
/// - `--key value`
/// - `--key=value`
pub fn parse_args() -> Result<TrainerConfig, String> {
    let mut cfg = TrainerConfig::default();

    let mut args = std::env::args().skip(1).peekable();

    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            return Err(usage());
        }

        let (k, v_opt) = split_key_value(&arg);

        match k.as_str() {
            "--eval" | "--eval-egev2" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.eval_egev2_path = PathBuf::from(v);
            }
            "--train-dir" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.train_dir = PathBuf::from(v);
            }
            "--val-dir" | "--validation-dir" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.val_dir = PathBuf::from(v);
            }
            "--out" | "--out-zip" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.out_zip = PathBuf::from(v);
            }
            "--epochs" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.epochs = v.parse().map_err(|_| format!("invalid --epochs: '{v}'"))?;
            }
            "--batch" | "--batch-size" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.batch_size = v.parse().map_err(|_| format!("invalid --batch-size: '{v}'"))?;
            }
            "--teacher-url" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.teacher_url = Some(v);
            }
            "--teacher-batch" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.teacher_batch_size = v
                    .parse()
                    .map_err(|_| format!("invalid --teacher-batch: '{v}'"))?;
            }
            "--lr" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.lr = v.parse().map_err(|_| format!("invalid --lr: '{v}'"))?;
            }
            "--lr-decay" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.lr_decay = v.parse().map_err(|_| format!("invalid --lr-decay: '{v}'"))?;
            }
            "--gen" | "--generation" => {
                let v = v_opt.or_else(|| args.next()).ok_or_else(|| format!("{k} requires a value"))?;
                cfg.gen = v.parse().map_err(|_| format!("invalid --gen: '{v}'"))?;
            }
            "--strict" => {
                cfg.strict = true;
            }
            "--quiet" => {
                cfg.quiet = true;
            }
            unknown => {
                return Err(format!("unknown arg: {unknown}\n\n{}", usage()));
            }
        }
    }

    Ok(cfg)
}

fn split_key_value(arg: &str) -> (String, Option<String>) {
    if let Some((k, v)) = arg.split_once('=') {
        (k.to_string(), Some(v.to_string()))
    } else {
        (arg.to_string(), None)
    }
}

pub fn usage() -> String {
    let d = TrainerConfig::default();
    format!(
        "sonetto_trainer (native CLI)\n\nUSAGE:\n  sonetto_trainer [options]\n\nOPTIONS:\n  --eval <path>            Path to eval.egev2 (default: {eval})\n  --train-dir <dir>        Training dataset dir (default: {train})\n  --val-dir <dir>          Validation transcript dir (default: {val})\n  --out <path>             Output zip path (default: {out})\n  --epochs <n>             Epochs (default: {epochs})\n  --batch-size <n>         Batch size for streaming (default: {batch})\n  --teacher-url <url>      Optional teacher base URL\n  --teacher-batch <n>      Teacher batch size (default: {tbatch})\n  --lr <float>             Initial learning rate (default: {lr})\n  --lr-decay <float>       LR decay factor (default: {decay})\n  --gen <n>                Generation counter (default: {gen})\n  --strict                 Fail on first parse error\n  --quiet                  Suppress stdout (still writes logs into zip)\n  -h, --help               Show this help\n",
        eval = d.eval_egev2_path.display(),
        train = d.train_dir.display(),
        val = d.val_dir.display(),
        out = d.out_zip.display(),
        epochs = d.epochs,
        batch = d.batch_size,
        tbatch = d.teacher_batch_size,
        lr = d.lr,
        decay = d.lr_decay,
        gen = d.gen,
    )
}
