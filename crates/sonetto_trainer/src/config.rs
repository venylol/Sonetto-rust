use std::path::PathBuf;

/// Trainer configuration parsed from CLI args.
///
/// Constraints:
/// - Must be runnable as a native CLI (no browser/WASM).
/// - Keep memory usage bounded: datasets are streamed and processed in batches.
/// - Maintain compatibility with the historical "teacher" HTTP protocol.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Path to an existing `eval.egev2` file.
    pub eval_egev2_path: PathBuf,

    /// Training dataset directory.
    /// Expected layout: `web/datasets/training/*.txt`.
    pub train_dir: PathBuf,

    /// Validation transcript directory.
    /// Expected layout: `web/datasets/validation/*.txt`.
    pub val_dir: PathBuf,

    /// Output ZIP path. The ZIP will contain at least:
    /// - eval.egev2
    /// - manifest.json
    /// - logs/trainer.log
    pub out_zip: PathBuf,

    /// Number of training epochs to run.
    pub epochs: usize,

    /// Batch size for streaming dataset processing.
    pub batch_size: usize,

    /// Base URL of the teacher server, e.g. http://127.0.0.1:8081
    pub teacher_url: Option<String>,

    /// Maximum number of teacher queries to batch in a single HTTP call.
    pub teacher_batch_size: usize,

    /// Initial learning rate (even if we are not applying real updates yet, we keep the schedule).
    pub lr: f32,

    /// Learning rate decay factor applied in the `lr_decay` stage.
    pub lr_decay: f32,

    /// Generation counter to store into manifest.json.
    pub gen: u64,

    /// If true, treat any parse error as fatal.
    /// If false, skip bad lines and continue.
    pub strict: bool,

    /// If true, suppress stdout logs (still written to the exported ZIP log).
    pub quiet: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            eval_egev2_path: PathBuf::from("eval.egev2"),
            train_dir: PathBuf::from("web/datasets/training"),
            val_dir: PathBuf::from("web/datasets/validation"),
            out_zip: PathBuf::from("out/Sonetto_Gen0_weights-only.zip"),
            epochs: 1,
            batch_size: 256,
            teacher_url: None,
            teacher_batch_size: 64,
            lr: 0.01,
            lr_decay: 0.5,
            gen: 0,
            strict: false,
            quiet: false,
        }
    }
}
