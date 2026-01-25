use serde_json::Value;
use std::fmt;
use std::time::Duration;

/// A single teacher query: evaluate a set of candidate moves on a position.
///
/// NOTE: "moves" are encoded as the historical Egaroucid-style policy indices:
/// `policy = 63 - bitpos`.
#[derive(Debug, Clone)]
pub struct TeacherQuery {
    pub board: String,
    pub moves: Vec<u8>,
}

#[derive(Debug)]
pub enum TeacherError {
    HttpStatus(u16, String),
    Transport(String),
    Json(String),
    Io(String),
    Protocol(String),
}

impl fmt::Display for TeacherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TeacherError::HttpStatus(code, body) => write!(f, "HTTP {code}: {body}"),
            TeacherError::Transport(e) => write!(f, "transport: {e}"),
            TeacherError::Json(e) => write!(f, "json: {e}"),
            TeacherError::Io(e) => write!(f, "io: {e}"),
            TeacherError::Protocol(e) => write!(f, "protocol: {e}"),
        }
    }
}

impl std::error::Error for TeacherError {}

/// Teacher server client.
///
/// Compatibility goals:
/// - Prefer `POST /get_teacher_guidance`.
/// - Fallback to `POST /eval_moves`.
///
/// The request/response formats are implemented to match historical Sonetto
/// worker behavior as closely as possible while remaining tolerant to minor
/// server-side JSON envelope differences.
#[derive(Clone)]
pub struct TeacherClient {
    base_url: String,
    agent: ureq::Agent,
}

impl TeacherClient {
    pub fn new(base_url: String) -> Self {
        // Keep timeouts finite so the CLI doesn't hang forever.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout_read(Duration::from_secs(60))
            .timeout_write(Duration::from_secs(60))
            .build();

        Self { base_url, agent }
    }

    /// Batch request using `/get_teacher_guidance`.
    ///
    /// Returns `Vec<Vec<i32>>` where each inner vector corresponds to the query's move list.
    pub fn get_teacher_guidance(&self, batch: &[TeacherQuery]) -> Result<Vec<Vec<i32>>, TeacherError> {
        let positions: Vec<Value> = batch
            .iter()
            .map(|q| {
                Value::Object(
                    [
                        ("board".to_string(), Value::String(q.board.clone())),
                        (
                            "moves".to_string(),
                            Value::Array(q.moves.iter().map(|&m| Value::from(m as u64)).collect()),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            })
            .collect();

        let req = Value::Object(
            [
                ("positions".to_string(), Value::Array(positions)),
            ]
            .into_iter()
            .collect(),
        );

        let v = self.post_json_value("/get_teacher_guidance", &req)?;

        // Accepted response envelopes:
        // - { results: [ { evals: [...] }, ... ] }
        // - { evaluations: [ [...], ... ] }
        // - { evals: [ [...], ... ] }
        if let Some(arr) = v.get("results").and_then(|x| x.as_array()) {
            let mut out: Vec<Vec<i32>> = Vec::with_capacity(arr.len());
            for (i, item) in arr.iter().enumerate() {
                let evals = item
                    .get("evals")
                    .or_else(|| item.get("evaluations"))
                    .or_else(|| item.get("values"))
                    .ok_or_else(|| TeacherError::Protocol(format!("missing evals in results[{i}]")))?;
                out.push(parse_i32_array(evals).ok_or_else(|| {
                    TeacherError::Protocol(format!("invalid eval array in results[{i}]"))
                })?);
            }
            return Ok(out);
        }

        if let Some(arr2) = v.get("evaluations").and_then(|x| x.as_array()) {
            let mut out: Vec<Vec<i32>> = Vec::with_capacity(arr2.len());
            for (i, item) in arr2.iter().enumerate() {
                out.push(parse_i32_array(item).ok_or_else(|| {
                    TeacherError::Protocol(format!("invalid evaluations[{i}] array"))
                })?);
            }
            return Ok(out);
        }

        if let Some(arr3) = v.get("evals").and_then(|x| x.as_array()) {
            // Could be nested, could be flat. Prefer nested.
            if arr3.iter().all(|x| x.is_array()) {
                let mut out: Vec<Vec<i32>> = Vec::with_capacity(arr3.len());
                for (i, item) in arr3.iter().enumerate() {
                    out.push(parse_i32_array(item).ok_or_else(|| {
                        TeacherError::Protocol(format!("invalid evals[{i}] array"))
                    })?);
                }
                return Ok(out);
            }
        }

        Err(TeacherError::Protocol(
            "unrecognized /get_teacher_guidance response envelope".to_string(),
        ))
    }

    /// Single-position request using `/eval_moves`.
    pub fn eval_moves(&self, q: &TeacherQuery) -> Result<Vec<i32>, TeacherError> {
        let req = Value::Object(
            [
                ("board".to_string(), Value::String(q.board.clone())),
                (
                    "moves".to_string(),
                    Value::Array(q.moves.iter().map(|&m| Value::from(m as u64)).collect()),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let v = self.post_json_value("/eval_moves", &req)?;

        let evals = v
            .get("evals")
            .or_else(|| v.get("evaluations"))
            .or_else(|| v.get("values"))
            .ok_or_else(|| TeacherError::Protocol("missing evals in /eval_moves".to_string()))?;

        parse_i32_array(evals)
            .ok_or_else(|| TeacherError::Protocol("invalid evals array in /eval_moves".to_string()))
    }

    /// Convenience wrapper that tries `/get_teacher_guidance` first and falls back to `/eval_moves`.
    pub fn eval_moves_batch(&self, batch: &[TeacherQuery]) -> Result<Vec<Vec<i32>>, TeacherError> {
        match self.get_teacher_guidance(batch) {
            Ok(v) => Ok(v),
            Err(_) => {
                // Fallback: call `/eval_moves` per position.
                let mut out: Vec<Vec<i32>> = Vec::with_capacity(batch.len());
                for q in batch {
                    out.push(self.eval_moves(q)?);
                }
                Ok(out)
            }
        }
    }

    fn post_json_value(&self, path: &str, body: &Value) -> Result<Value, TeacherError> {
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );

        let body_str = serde_json::to_string(body).map_err(|e| TeacherError::Json(e.to_string()))?;

        let resp = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body_str);

        match resp {
            Ok(r) => {
                let text = r
                    .into_string()
                    .map_err(|e| TeacherError::Io(e.to_string()))?;
                serde_json::from_str(&text).map_err(|e| TeacherError::Json(e.to_string()))
            }
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_else(|_| String::new());
                Err(TeacherError::HttpStatus(code, text))
            }
            Err(ureq::Error::Transport(t)) => Err(TeacherError::Transport(t.to_string())),
        }
    }
}

fn parse_i32_array(v: &Value) -> Option<Vec<i32>> {
    let arr = v.as_array()?;
    let mut out: Vec<i32> = Vec::with_capacity(arr.len());
    for x in arr {
        if let Some(n) = x.as_i64() {
            out.push(n as i32);
        } else if let Some(n) = x.as_u64() {
            out.push(n as i32);
        } else if let Some(f) = x.as_f64() {
            out.push(f.round() as i32);
        } else {
            return None;
        }
    }
    Some(out)
}
