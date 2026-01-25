//! Egaroucid `eval.egev2` compression format.
//!
//! This is a very small run-length encoding (RLE) specifically for **runs of zeros**.
//!
//! File layout (as produced by Egaroucid's `output_egev2.cpp`):
//!
//! - `u32`/`int32` little-endian: number of **compressed** `i16` elements, `n`.
//! - `n` × `i16` little-endian: compressed stream.
//!
//! Stream element semantics (as used by `load_unzip_egev2`):
//!
//! - If `x >= N_ZEROS_PLUS (4096)`: expand to `(x - N_ZEROS_PLUS)` zeros.
//! - Else: `x` is a literal parameter value.
//!
//! Encoding rule:
//!
//! - Literal zeros are never written; instead, a run of `k` consecutive zeros is written as a single
//!   element `(N_ZEROS_PLUS + k)`.
//! - Runs are split so that `k <= N_MAX_ZEROS (28600)`.
//!
//! Notes:
//! - Egaroucid's exporter also clamps each literal parameter to `[-EVAL_MAX, EVAL_MAX]` so that
//!   no literal value collides with the RLE marker region (`>= 4096`).

use core::cmp::Ordering;

/// Marker offset. Values `>= N_ZEROS_PLUS` represent a run of zeros.
pub const N_ZEROS_PLUS: i16 = 1 << 12; // 4096

/// Maximum length of a single zero run written as one element.
pub const N_MAX_ZEROS: usize = 28_600;

/// Maximum absolute literal evaluation value used by Egaroucid (16 patterns).
pub const EVAL_MAX: i16 = 4091;

/// Decode a full `eval.egev2` file into the uncompressed parameter vector.
///
/// This function is intentionally **panic-free** and returns an empty vector on malformed input.
pub fn decode_egev2(bytes: &[u8]) -> Vec<i16> {
    match decode_egev2_impl(bytes) {
        Ok(v) => v,
        Err(_) => Vec::new(),
    }
}

fn decode_egev2_impl(bytes: &[u8]) -> Result<Vec<i16>, ()> {
    if bytes.len() < 4 {
        return Err(());
    }

    let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

    // Need at least 2*n bytes after the 4-byte header.
    let need = 4usize
        .checked_add(n.checked_mul(2).ok_or(())?)
        .ok_or(())?;
    if bytes.len() < need {
        return Err(());
    }

    let mut out: Vec<i16> = Vec::with_capacity(n);
    let mut off = 4usize;

    for _ in 0..n {
        let v = i16::from_le_bytes([bytes[off], bytes[off + 1]]);
        if v >= N_ZEROS_PLUS {
            let k = (v as i32 - N_ZEROS_PLUS as i32) as usize;
            let new_len = out.len().checked_add(k).ok_or(())?;
            out.resize(new_len, 0i16);
        } else {
            out.push(v);
        }
        off += 2;
    }

    Ok(out)
}

/// Encode an uncompressed parameter vector into Egaroucid's `eval.egev2` bytes.
///
/// The output matches the canonical encoding produced by Egaroucid's `output_egev2.cpp`:
/// - zero runs are emitted as `N_ZEROS_PLUS + run_len` and split at `N_MAX_ZEROS`.
/// - non-zero literals are (optionally) clamped to `[-EVAL_MAX, EVAL_MAX]`.
pub fn encode_egev2(params: &[i16]) -> Vec<u8> {
    let mut compressed: Vec<i16> = Vec::new();
    let mut run_zeros: usize = 0;

    for &raw in params {
        let v = clamp_i16(raw, -EVAL_MAX, EVAL_MAX);
        if v == 0 {
            run_zeros += 1;
            if run_zeros == N_MAX_ZEROS {
                compressed.push((N_ZEROS_PLUS as i32 + run_zeros as i32) as i16);
                run_zeros = 0;
            }
        } else {
            if run_zeros > 0 {
                compressed.push((N_ZEROS_PLUS as i32 + run_zeros as i32) as i16);
                run_zeros = 0;
            }
            compressed.push(v);
        }
    }

    if run_zeros > 0 {
        compressed.push((N_ZEROS_PLUS as i32 + run_zeros as i32) as i16);
    }

    // Serialize
    let mut out: Vec<u8> = Vec::with_capacity(4 + compressed.len() * 2);
    let n = compressed.len() as u32;
    out.extend_from_slice(&n.to_le_bytes());
    for v in compressed {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[inline(always)]
fn clamp_i16(x: i16, lo: i16, hi: i16) -> i16 {
    match x.cmp(&lo) {
        Ordering::Less => lo,
        Ordering::Equal | Ordering::Greater => match x.cmp(&hi) {
            Ordering::Greater => hi,
            _ => x,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_egev2, encode_egev2, EVAL_MAX, N_MAX_ZEROS, N_ZEROS_PLUS};

    fn read_compressed_stream(bytes: &[u8]) -> Vec<i16> {
        assert!(bytes.len() >= 4, "missing u32 header");
        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert_eq!(bytes.len(), 4 + n * 2, "byte length does not match header count");

        let mut out: Vec<i16> = Vec::with_capacity(n);
        let mut off = 4usize;
        for _ in 0..n {
            out.push(i16::from_le_bytes([bytes[off], bytes[off + 1]]));
            off += 2;
        }
        out
    }

    #[test]
    fn encode_decode_roundtrip_smoke() {
        let params: Vec<i16> = vec![0, 0, 1, -2, 0, 0, 0, 5, 0, -7, 0];
        let bytes = encode_egev2(&params);
        let decoded = decode_egev2(&bytes);
        assert_eq!(decoded, params);
    }

    #[test]
    fn encode_never_emits_literal_zero() {
        let params: Vec<i16> = vec![0, 0, 0, 1, 0, 2, 0, 0];
        let bytes = encode_egev2(&params);
        let stream = read_compressed_stream(&bytes);
        assert!(
            !stream.iter().any(|&v| v == 0),
            "compressed stream must not contain literal zeros"
        );
    }

    #[test]
    fn encode_splits_long_zero_runs() {
        let params: Vec<i16> = vec![0; N_MAX_ZEROS + 10];
        let bytes = encode_egev2(&params);
        let stream = read_compressed_stream(&bytes);

        assert_eq!(stream.len(), 2, "expected split into two run markers");
        assert_eq!(stream[0] as i32, N_ZEROS_PLUS as i32 + N_MAX_ZEROS as i32);
        assert_eq!(stream[1] as i32, N_ZEROS_PLUS as i32 + 10);

        // roundtrip
        let decoded = decode_egev2(&bytes);
        assert_eq!(decoded, params);
    }

    #[test]
    fn encode_clamps_literals_into_safe_range() {
        // Values outside [-EVAL_MAX, EVAL_MAX] must be clamped to avoid colliding with
        // the run-marker region (>= 4096).
        let params: Vec<i16> = vec![5000, -5000, EVAL_MAX + 1, -(EVAL_MAX + 1)];
        let bytes = encode_egev2(&params);
        let decoded = decode_egev2(&bytes);
        assert_eq!(decoded, vec![EVAL_MAX, -EVAL_MAX, EVAL_MAX, -EVAL_MAX]);
    }
}
