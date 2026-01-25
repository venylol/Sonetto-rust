//! Sensei-style direct-mapped hash map (transposition cache).
//!
//! This module is a close port of OthelloSensei's `HashMap<bits>` used by
//! `EvaluatorAlphaBeta`.
//!
//! Key properties:
//! - **direct mapped** (one entry per bucket)
//! - **best-effort lock-free** access via a per-entry `busy` flag
//! - stores `(lower, upper)` bounds and the best + second-best move
//! - hash function matches Sensei's row-wise XOR over byte tables derived from a
//!   constexpr LCG (`utils/random.h` in the native code).
//!
//! Why a separate structure instead of reusing `crate::tt`?
//! - Sensei stores both bounds simultaneously and uses them with a slightly
//!   different cutoff condition. Keeping this separate reduces coupling and
//!   makes it easier to compare against the native implementation.

use crate::coord::{Move, PASS};
use crate::score::Score;

use core::cell::UnsafeCell;
use core::mem::size_of;
use std::sync::OnceLock;

#[cfg(target_has_atomic = "8")]
use core::sync::atomic::{AtomicBool, Ordering};

const LAST_ROW_MASK: u64 = 0xFF;

/// Sensei's score \"sentinel\" bounds.
/// (Native uses kMinEvalLarge/kMaxEvalLarge; our Score domain is wider, but
/// we use a large window consistent with the alpha-beta backend.)
pub const MIN_SCORE: Score = -1_000_000;
pub const MAX_SCORE: Score = 1_000_000;

/// Public, copied-out hash entry (mirrors Sensei's `HashMapEntry`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SenseiHashEntry {
    pub player: u64,
    pub opponent: u64,
    pub lower: Score,
    pub upper: Score,
    pub depth: u8,
    pub best_move: Move,
    pub second_best_move: Move,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EntryData {
    player: u64,
    opponent: u64,
    lower: Score,
    upper: Score,
    depth: u8,
    best_move: Move,
    second_best_move: Move,
    // Padding for alignment (mirrors the \"no inheritance\" trick in C++).
    _pad: u8,
}

impl Default for EntryData {
    fn default() -> Self {
        Self {
            player: 0,
            opponent: 0,
            lower: MIN_SCORE,
            upper: MAX_SCORE,
            depth: 0,
            best_move: PASS,
            second_best_move: PASS,
            _pad: 0,
        }
    }
}

struct EntryInternal {
    data: UnsafeCell<EntryData>,
    #[cfg(target_has_atomic = "8")]
    busy: AtomicBool,
    #[cfg(not(target_has_atomic = "8"))]
    busy: UnsafeCell<bool>,
}

// Safety: access is synchronized by `busy` (AtomicBool on atomic targets; on
// non-atomic targets we assume single-threaded execution).
unsafe impl Sync for EntryInternal {}
unsafe impl Send for EntryInternal {}

impl EntryInternal {
    #[inline(always)]
    fn new() -> Self {
        Self {
            data: UnsafeCell::new(EntryData::default()),
            #[cfg(target_has_atomic = "8")]
            busy: AtomicBool::new(false),
            #[cfg(not(target_has_atomic = "8"))]
            busy: UnsafeCell::new(false),
        }
    }

    #[inline(always)]
    fn try_lock(&self) -> bool {
        #[cfg(target_has_atomic = "8")]
        {
            self.busy
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        }
        #[cfg(not(target_has_atomic = "8"))]
        unsafe {
            let b = &mut *self.busy.get();
            if *b {
                false
            } else {
                *b = true;
                true
            }
        }
    }

    #[inline(always)]
    fn unlock(&self) {
        #[cfg(target_has_atomic = "8")]
        self.busy.store(false, Ordering::Release);

        #[cfg(not(target_has_atomic = "8"))]
        unsafe {
            *self.busy.get() = false;
        }
    }

    #[inline(always)]
    unsafe fn read_data(&self) -> EntryData {
        *self.data.get()
    }

    #[inline(always)]
    unsafe fn write_data(&self, d: EntryData) {
        *self.data.get() = d;
    }
}

/// Hash tables computed identically to Sensei's `constexpr HashValues`.
struct HashTables {
    // Layout: [bits][row][byte] flattened.
    player: Vec<u32>,
    opponent: Vec<u32>,
}

impl HashTables {
    const BITS: usize = 33;
    const ROWS: usize = 8;
    const BYTES: usize = 256;

    #[inline(always)]
    fn idx(bits: usize, row: usize, byte: usize) -> usize {
        (bits * Self::ROWS + row) * Self::BYTES + byte
    }

    fn new() -> Self {
        // Reproduce Sensei's constexpr LCG.
        let mut last: u32 = 1_940_869_496;
        let a: u32 = 2_891_336_453;
        let c: u32 = 7;

        let mut next = || {
            last = last.wrapping_mul(a).wrapping_add(c);
            last
        };

        let n = Self::BITS * Self::ROWS * Self::BYTES;
        let mut player = vec![0u32; n];
        let mut opponent = vec![0u32; n];

        for row in 0..Self::ROWS {
            for byte in 0..Self::BYTES {
                let p = next();
                let o = next();

                for bits in 0..32usize {
                    let m = 1u32 << bits;
                    player[Self::idx(bits, row, byte)] = p % m;
                    opponent[Self::idx(bits, row, byte)] = o % m;
                }
                player[Self::idx(32, row, byte)] = p;
                opponent[Self::idx(32, row, byte)] = o;
            }
        }

        Self { player, opponent }
    }

    #[inline(always)]
    fn p(&self, bits: usize, row: usize, byte: u8) -> u32 {
        self.player[Self::idx(bits, row, byte as usize)]
    }

    #[inline(always)]
    fn o(&self, bits: usize, row: usize, byte: u8) -> u32 {
        self.opponent[Self::idx(bits, row, byte as usize)]
    }
}

fn tables() -> &'static HashTables {
    static T: OnceLock<HashTables> = OnceLock::new();
    T.get_or_init(HashTables::new)
}

#[inline(always)]
fn hash_bits(bits: u8, player: u64, opponent: u64) -> u32 {
    debug_assert!(bits <= 32);
    let bits_usize = bits as usize;
    let t = tables();

    // Sensei treats each 8-bit chunk as a \"row\" (little-endian).
    let p0 = (player & LAST_ROW_MASK) as u8;
    let p1 = ((player >> 8) & LAST_ROW_MASK) as u8;
    let p2 = ((player >> 16) & LAST_ROW_MASK) as u8;
    let p3 = ((player >> 24) & LAST_ROW_MASK) as u8;
    let p4 = ((player >> 32) & LAST_ROW_MASK) as u8;
    let p5 = ((player >> 40) & LAST_ROW_MASK) as u8;
    let p6 = ((player >> 48) & LAST_ROW_MASK) as u8;
    let p7 = (player >> 56) as u8;

    let o0 = (opponent & LAST_ROW_MASK) as u8;
    let o1 = ((opponent >> 8) & LAST_ROW_MASK) as u8;
    let o2 = ((opponent >> 16) & LAST_ROW_MASK) as u8;
    let o3 = ((opponent >> 24) & LAST_ROW_MASK) as u8;
    let o4 = ((opponent >> 32) & LAST_ROW_MASK) as u8;
    let o5 = ((opponent >> 40) & LAST_ROW_MASK) as u8;
    let o6 = ((opponent >> 48) & LAST_ROW_MASK) as u8;
    let o7 = (opponent >> 56) as u8;

    t.p(bits_usize, 0, p0)
        ^ t.p(bits_usize, 1, p1)
        ^ t.p(bits_usize, 2, p2)
        ^ t.p(bits_usize, 3, p3)
        ^ t.p(bits_usize, 4, p4)
        ^ t.p(bits_usize, 5, p5)
        ^ t.p(bits_usize, 6, p6)
        ^ t.p(bits_usize, 7, p7)
        ^ t.o(bits_usize, 0, o0)
        ^ t.o(bits_usize, 1, o1)
        ^ t.o(bits_usize, 2, o2)
        ^ t.o(bits_usize, 3, o3)
        ^ t.o(bits_usize, 4, o4)
        ^ t.o(bits_usize, 5, o5)
        ^ t.o(bits_usize, 6, o6)
        ^ t.o(bits_usize, 7, o7)
}

/// Direct-mapped Sensei hash map.
pub struct SenseiHashMap {
    bits: u8,
    mask: u32,
    slots: Vec<EntryInternal>,
}

unsafe impl Sync for SenseiHashMap {}
unsafe impl Send for SenseiHashMap {}

impl SenseiHashMap {
    /// Create a new map sized to approximately `hash_size_mb`.
    ///
    /// The actual size is rounded down to the nearest power of two.
    pub fn new_mb(hash_size_mb: usize) -> Self {
        let bytes = hash_size_mb.max(1) * 1024 * 1024;
        let entry_bytes = size_of::<EntryInternal>().max(1);
        let n = (bytes / entry_bytes).max(1);

        // Sensei sizes the table as 1<<bits; we pick the largest power-of-two
        // that fits in the requested budget.
        let floor_bits = (usize::BITS - 1 - n.leading_zeros()) as u8;

        // Clamp to 2^32 so we can reuse the native hash table.
        let bits = floor_bits.min(32);

        let size = 1usize << bits;
        let mut slots = Vec::with_capacity(size);
        for _ in 0..size {
            slots.push(EntryInternal::new());
        }

        Self {
            bits,
            mask: (size as u32) - 1,
            slots,
        }
    }

    #[inline(always)]
    fn index(&self, player: u64, opponent: u64) -> usize {
        let h = hash_bits(self.bits, player, opponent);
        (h & self.mask) as usize
    }

    /// Reset the table (best-effort; clears keys).
    ///
    /// This mirrors Sensei's `Reset()`. It is intentionally O(N).
    pub fn reset(&mut self) {
        for e in &self.slots {
            if e.try_lock() {
                unsafe {
                    e.write_data(EntryData::default());
                }
                e.unlock();
            }
        }
    }

    /// Get an entry for `(player, opponent)` if present.
    ///
    /// Returns a *copy* (Sensei returns a heap-allocated copy for the same reason:
    /// concurrent writers might invalidate a reference).
    #[inline(always)]
    pub fn get(&self, player: u64, opponent: u64) -> Option<SenseiHashEntry> {
        let idx = self.index(player, opponent);
        let slot = &self.slots[idx];

        if !slot.try_lock() {
            return None;
        }

        let d = unsafe { slot.read_data() };
        if d.player != player || d.opponent != opponent {
            slot.unlock();
            return None;
        }

        let out = SenseiHashEntry {
            player: d.player,
            opponent: d.opponent,
            lower: d.lower,
            upper: d.upper,
            depth: d.depth,
            best_move: d.best_move,
            second_best_move: d.second_best_move,
        };

        slot.unlock();
        Some(out)
    }

    /// Update the entry for `(player, opponent)`.
    ///
    /// Bounds are stored exactly like Sensei:
    /// - `stored_lower = eval` if `eval > lower` else `MIN_SCORE`
    /// - `stored_upper = eval` if `eval < upper` else `MAX_SCORE`
    #[inline(always)]
    pub fn update(
        &self,
        player: u64,
        opponent: u64,
        depth: u8,
        eval: Score,
        lower: Score,
        upper: Score,
        best_move: Move,
        second_best_move: Move,
    ) {
        let idx = self.index(player, opponent);
        let slot = &self.slots[idx];

        if !slot.try_lock() {
            return;
        }

        let stored_lower = if eval > lower { eval } else { MIN_SCORE };
        let stored_upper = if eval < upper { eval } else { MAX_SCORE };

        let d = EntryData {
            player,
            opponent,
            lower: stored_lower,
            upper: stored_upper,
            depth,
            best_move,
            second_best_move,
            _pad: 0,
        };

        unsafe { slot.write_data(d) };
        slot.unlock();
    }
}