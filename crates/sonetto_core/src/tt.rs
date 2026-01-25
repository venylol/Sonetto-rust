//! Transposition table (TT): lock-free cluster table (atomic or single-thread plain build).
//!
//! Entry fields (per spec):
//! - key: u64
//! - depth: u8
//! - flag: Bound
//! - value: Score
//! - best_move: u8 (Move)
//! - best_move2: Option<u8>

#[cfg(all(feature = "tt_atomic", feature = "tt_plain"))]
compile_error!("features `tt_atomic` and `tt_plain` are mutually exclusive");

#[cfg(feature = "tt_atomic")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "tt_plain")]
use std::cell::Cell;

#[cfg(feature = "tt_atomic")]
#[derive(Default)]
struct SlotWord(AtomicU64);

#[cfg(feature = "tt_plain")]
#[derive(Default)]
struct SlotWord(Cell<u64>);

#[cfg(feature = "tt_atomic")]
impl SlotWord {
    #[inline(always)]
    fn new(v: u64) -> Self {
        Self(AtomicU64::new(v))
    }

    #[inline(always)]
    fn load_acquire(&self) -> u64 {
        // PERF: use Relaxed ordering and rely on the embedded `key_tag` as a
        // torn-read filter.
        //
        // In the atomic TT, keys and data are stored in two independent atomics.
        // Readers may observe mismatched pairs transiently; we detect and reject
        // those by checking the 16-bit tag packed into `data`.
        //
        // Acquire/Release provides stronger ordering but measurably lowers NPS.
        // Relaxed keeps the TT lock-free and safe (TT is a heuristic cache), while
        // increasing throughput.
        self.0.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn store_release(&self, v: u64) {
        // See `load_acquire` comment.
        self.0.store(v, Ordering::Relaxed)
    }

    #[inline(always)]
    fn store_relaxed(&self, v: u64) {
        self.0.store(v, Ordering::Relaxed)
    }
}

#[cfg(feature = "tt_plain")]
impl SlotWord {
    #[inline(always)]
    fn new(v: u64) -> Self {
        Self(Cell::new(v))
    }

    #[inline(always)]
    fn load_acquire(&self) -> u64 {
        self.0.get()
    }

    #[inline(always)]
    fn store_release(&self, v: u64) {
        self.0.set(v)
    }

    #[inline(always)]
    fn store_relaxed(&self, v: u64) {
        self.0.set(v)
    }
}

use crate::coord::{Move, PASS};
use crate::score::Score;

/// TT bound flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    Exact = 0,
    Lower = 1,
    Upper = 2,
}

/// Public, decoded TT entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TTEntry {
    pub key: u64,
    pub depth: u8,
    pub flag: Bound,
    pub value: Score,
    pub best_move: Move,
    pub best_move2: Option<Move>,
}

const CLUSTER_SIZE: usize = 4;

// Packed data layout (u64):
// bits  0..=15 : value (i16 bits; Score is clamped to i16 range)
// bits 16..=23 : depth (u8)
// bits 24..=25 : flag (2 bits)
// bits 26..=33 : best_move (u8)
// bits 34..=41 : best_move2 (u8; PASS means None)
// bits 42..=57 : key_tag (u16; integrity tag to avoid torn (key,data) reads)
// bits 58..=62 : reserved
// bit      63  : valid
//
// Why `key_tag`?
// This TT stores `key` and `data` in two independent atomics. On weak memory
// models (and/or with compiler reordering), readers may transiently observe a
// mismatched pair (old key with new data or vice versa). A 16-bit tag derived
// from the key is packed into `data` so we can reject such torn reads cheaply.
const SHIFT_DEPTH: u64 = 16;
const SHIFT_FLAG: u64 = 24;
const SHIFT_BM1: u64 = 26;
const SHIFT_BM2: u64 = 34;
const SHIFT_TAG: u64 = 42;

const MASK_U8: u64 = 0xFF;
const MASK_U16: u64 = 0xFFFF;
const MASK_FLAG: u64 = 0x3;

const VALID_BIT: u64 = 1u64 << 63;

#[inline(always)]
fn key_tag(key: u64) -> u16 {
    // Lightweight reversible mix; tag is used only as a torn-read filter.
    let x = key ^ (key >> 32);
    let x = x ^ (x >> 16);
    (x & 0xFFFF) as u16
}

#[inline(always)]
fn pack_data(
    depth: u8,
    flag: Bound,
    value: Score,
    best_move: Move,
    best_move2: Option<Move>,
    tag: u16,
) -> u64 {
    // Score domain is bounded by the Othello disc-diff scale (±2048), so i16 is safe.
    // Still clamp defensively to avoid catastrophic wrap on accidental out-of-range values.
    let v_clamped = value.clamp(i16::MIN as i32, i16::MAX as i32);
    debug_assert_eq!(v_clamped, value, "TT.store: Score out of i16 range: {value}");
    let v = (v_clamped as i16 as u16) as u64;

    let d = (depth as u64) << SHIFT_DEPTH;
    let f = (flag as u64) << SHIFT_FLAG;
    let bm1 = (best_move as u64) << SHIFT_BM1;
    let bm2v = best_move2.unwrap_or(PASS);
    let bm2 = (bm2v as u64) << SHIFT_BM2;
    let t = (tag as u64) << SHIFT_TAG;

    VALID_BIT | v | d | f | bm1 | bm2 | t
}

#[inline(always)]
fn unpack_data(key: u64, data: u64) -> Option<TTEntry> {
    if (data & VALID_BIT) == 0 {
        return None;
    }
    let v = (data & MASK_U16) as u16;
    let value = (v as i16) as i32;

    let depth = ((data >> SHIFT_DEPTH) & MASK_U8) as u8;
    let flag_bits = ((data >> SHIFT_FLAG) & MASK_FLAG) as u8;
    let flag = match flag_bits {
        0 => Bound::Exact,
        1 => Bound::Lower,
        2 => Bound::Upper,
        _ => Bound::Exact,
    };

    let best_move = ((data >> SHIFT_BM1) & MASK_U8) as u8;
    let bm2 = ((data >> SHIFT_BM2) & MASK_U8) as u8;
    let best_move2 = if bm2 == PASS { None } else { Some(bm2) };

    Some(TTEntry {
        key,
        depth,
        flag,
        value,
        best_move,
        best_move2,
    })
}

struct TTSlot {
    key: SlotWord,
    data: SlotWord,
}

impl TTSlot {
    #[inline(always)]
    fn new() -> Self {
        Self {
            key: SlotWord::new(0),
            data: SlotWord::new(0),
        }
    }

    #[inline(always)]
    fn load_key(&self) -> u64 {
        self.key.load_acquire()
    }

    #[inline(always)]
    fn load_data(&self) -> u64 {
        self.data.load_acquire()
    }

    #[inline(always)]
    fn store(&self, key: u64, data: u64) {
        // Write data first, then publish key.
        self.data.store_release(data);
        self.key.store_release(key);
    }

    #[inline(always)]
    fn clear(&self) {
        self.data.store_relaxed(0);
        self.key.store_relaxed(0);
    }
}

#[repr(align(64))]
struct TTCluster {
    slots: [TTSlot; CLUSTER_SIZE],
}

impl TTCluster {
    #[inline(always)]
    fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| TTSlot::new()),
        }
    }
}

/// Lock-free transposition table.
///
/// - Clustered (4-way) buckets
/// - Replacement: prefer empty; otherwise replace smallest depth
/// - Concurrency: atomic build is Send+Sync; plain build is single-threaded (not Sync)
pub struct TranspositionTable {
    clusters: Vec<TTCluster>,
    mask: usize,
}

impl TranspositionTable {
    /// Create TT by size in **megabytes**.
    ///
    /// Table uses 4-way clusters; each slot stores (key,data) as 16 bytes.
    pub fn new(megabytes: usize) -> Self {
        let bytes = megabytes.saturating_mul(1024 * 1024);
        let cluster_bytes = CLUSTER_SIZE * (8 + 8); // key u64 + data u64
        let mut clusters = bytes / cluster_bytes;
        if clusters < 1 {
            clusters = 1;
        }
        // round down to power-of-two
        clusters = floor_pow2(clusters).max(1);

        let mut v = Vec::with_capacity(clusters);
        for _ in 0..clusters {
            v.push(TTCluster::new());
        }

        Self {
            clusters: v,
            mask: clusters - 1,
        }
    }

    #[inline(always)]
    fn idx(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    /// Prefetch the TT cluster for `key` into cache (best-effort).
    ///
    /// This is a pure performance hint (ported from common Othello engines).
    /// It must never affect correctness.
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let idx = self.idx(key);
            let ptr = &self.clusters[idx] as *const TTCluster as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = key;
        }
    }

    /// Probe TT by full key.
    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let c = &self.clusters[self.idx(key)];
        // In the atomic build, verify the embedded key tag to reject torn reads.
        // In the plain build, key/data are coherent, so we can skip this check.
        let expected_tag: u16 = if cfg!(feature = "tt_atomic") {
            key_tag(key)
        } else {
            0
        };
        for s in &c.slots {
            let k = s.load_key();
            if k != key {
                continue;
            }
            let d = s.load_data();
            if cfg!(feature = "tt_atomic") {
                // Guard against torn (key,data) reads in concurrent scenarios.
                let tag = ((d >> SHIFT_TAG) & MASK_U16) as u16;
                if tag != expected_tag {
                    continue;
                }
            }
            if let Some(e) = unpack_data(k, d) {
                return Some(e);
            }
        }
        None
    }

    /// Store entry into TT.
    ///
    /// `best_move2` is optional; store `PASS` sentinel when None.
    #[inline(always)]
    pub fn store(
        &self,
        key: u64,
        depth: u8,
        flag: Bound,
        value: Score,
        best_move: Move,
        best_move2: Option<Move>,
    ) {
        let c = &self.clusters[self.idx(key)];
        // Only compute/store the tag when it matters (atomic build).
        let tag: u16 = if cfg!(feature = "tt_atomic") { key_tag(key) } else { 0 };

        // 1) Replace same-key slot (prefer deeper/equal depth overwrite).
        for s in &c.slots {
            let k = s.load_key();
            if k != key {
                continue;
            }
            let old = s.load_data();
            let old_depth = ((old >> SHIFT_DEPTH) & MASK_U8) as u8;
            if (old & VALID_BIT) == 0 || depth >= old_depth {
                let data = pack_data(depth, flag, value, best_move, best_move2, tag);
                s.store(key, data);
            }
            return;
        }

        // 2) Find empty slot.
        for s in &c.slots {
            if (s.load_data() & VALID_BIT) == 0 {
                let data = pack_data(depth, flag, value, best_move, best_move2, tag);
                s.store(key, data);
                return;
            }
        }

        // 3) Replace the shallowest slot (bucket strategy).
        let mut victim_i = 0usize;
        let mut victim_depth = u8::MAX;
        for (i, s) in c.slots.iter().enumerate() {
            let d = s.load_data();
            let dep = ((d >> SHIFT_DEPTH) & MASK_U8) as u8;
            if dep < victim_depth {
                victim_depth = dep;
                victim_i = i;
            }
        }

        let data = pack_data(depth, flag, value, best_move, best_move2, tag);
        c.slots[victim_i].store(key, data);
    }

    /// Clear all entries (useful for tests / new game).
    pub fn clear(&self) {
        for c in &self.clusters {
            for s in &c.slots {
                s.clear();
            }
        }
    }
}

#[inline(always)]
fn floor_pow2(mut x: usize) -> usize {
    if x == 0 {
        return 0;
    }
    // highest power of two <= x
    let lz = x.leading_zeros() as usize;
    let bits = usize::BITS as usize;
    let shift = bits - 1 - lz;
    x = 1usize << shift;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn tt_is_send() {
        assert_send::<TranspositionTable>();
    }

    #[cfg(feature = "tt_atomic")]
    fn assert_send_sync<T: Send + Sync>() {}

    #[cfg(feature = "tt_atomic")]
    #[test]
    fn tt_is_send_sync() {
        assert_send_sync::<TranspositionTable>();
    }

    #[test]
    fn tt_store_probe_roundtrip() {
        let tt = TranspositionTable::new(1);
        let key = 0x0123_4567_89AB_CDEFu64;

        tt.store(key, 7, Bound::Exact, 42, 19, Some(20));
        let e = tt.probe(key).expect("probe hit");
        assert_eq!(e.key, key);
        assert_eq!(e.depth, 7);
        assert_eq!(e.flag, Bound::Exact);
        assert_eq!(e.value, 42);
        assert_eq!(e.best_move, 19);
        assert_eq!(e.best_move2, Some(20));
    }

    #[test]
    fn tt_overwrite_same_key_prefers_deeper() {
        let tt = TranspositionTable::new(1);
        let key = 0xDEAD_BEEF_0000_0001u64;

        tt.store(key, 2, Bound::Upper, -10, 19, None);
        tt.store(key, 5, Bound::Exact, 123, 20, Some(26));

        let e = tt.probe(key).unwrap();
        assert_eq!(e.depth, 5);
        assert_eq!(e.flag, Bound::Exact);
        assert_eq!(e.value, 123);
        assert_eq!(e.best_move, 20);
        assert_eq!(e.best_move2, Some(26));
    }
}
