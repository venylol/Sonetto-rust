use std::time::{SystemTime, UNIX_EPOCH};

/// Very small logger that:
/// - records lines in-memory (so we can embed them into the exported ZIP)
/// - optionally mirrors to stdout
#[derive(Debug)]
pub struct RunLog {
    lines: Vec<String>,
    mirror_stdout: bool,
}

impl RunLog {
    pub fn new(mirror_stdout: bool) -> Self {
        Self {
            lines: Vec::new(),
            mirror_stdout,
        }
    }

    pub fn line(&mut self, msg: impl AsRef<str>) {
        let ts = utc_timestamp_s();
        let s = format!("[{ts}] {}", msg.as_ref());
        if self.mirror_stdout {
            println!("{s}");
        }
        self.lines.push(s);
    }

    pub fn raw(&mut self, msg: impl AsRef<str>) {
        let s = msg.as_ref().to_string();
        if self.mirror_stdout {
            println!("{s}");
        }
        self.lines.push(s);
    }

    pub fn into_string(self) -> String {
        let mut out = String::new();
        for l in self.lines {
            out.push_str(&l);
            out.push('\n');
        }
        out
    }
}

fn utc_timestamp_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
