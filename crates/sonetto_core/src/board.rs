//! Board representation + make/undo.
//!
//! ## P2-3: Player/Opponent representation
//!
//! Sonetto originally stored bitboards in **absolute** color order:
//! `bits[Black], bits[White]` plus `side`.
//!
//! Phase2-3 switches the board to the more common and faster **player/opponent**
//! representation:
//!
//! - `player`: discs of the **side-to-move**
//! - `opponent`: discs of the **non-moving** side
//! - `side`: the **absolute** color of `player` (Black or White)
//!
//! After each move (including PASS), we **swap** `(player, opponent)` and flip
//! `side`. This removes repeated `side.idx()` / `side.other()` indexing from hot
//! paths (movegen, flips, endgame solvers, etc.) while keeping the absolute color
//! available for:
//! - Zobrist hashing (which is keyed by absolute color)
//! - EGEV2 evaluation digit swapping (Black identity, White swapped)
//! - External I/O and tooling.

use crate::coord::{Move, PASS};
use crate::features::occ::OccMap;
use crate::features::update::{
    rollback_features_for_move as rollback_pattern_features_for_move,
    update_features_for_move as update_pattern_features_for_move,
};
use crate::flips::flips_for_move_unchecked;
use crate::zobrist;

/// Disc color.
///
/// NOTE: `Black as usize == 0`, `White as usize == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    White = 1,
}

impl Default for Color {
    #[inline(always)]
    fn default() -> Self {
        // Black is the canonical "zero" color in Sonetto (see repr/u8 and idx()).
        Color::Black
    }
}

impl Color {
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn other(self) -> Self {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    /// Absolute digit encoding for feature base-3 IDs.
    /// - empty = 0
    /// - black = 1
    /// - white = 2
    #[inline(always)]
    pub const fn digit_abs(self) -> u16 {
        match self {
            Color::Black => 1,
            Color::White => 2,
        }
    }
}

/// Undo information for a single make/undo.
///
/// We store only what is needed for a fast reversible update.
#[derive(Clone, Copy, Debug, Default)]
pub struct Undo {
    pub mv: Move,
    pub mv_bit: u64,
    pub flips: u64,

    pub old_hash: u64,
    pub old_empty: u8,
    pub old_side: Color,
}

/// Game board.
///
/// Required fields (engine contract):
/// - `player`, `opponent`
/// - `side`
/// - `empty_count`
/// - `hash`
/// - `feat_id_abs`
#[derive(Clone, Debug)]
pub struct Board {
    /// Bitboard of the side-to-move (player).
    pub player: u64,
    /// Bitboard of the non-moving side (opponent).
    pub opponent: u64,

    /// Absolute color of `player`.
    pub side: Color,

    pub empty_count: u8,
    pub hash: u64,

    /// Whether `feat_id_abs` currently contains **EGEV2 absolute ternary pattern IDs**.
    ///
    /// This flag exists because both the legacy per-square digit cache and the
    /// EGEV2 pattern-ID cache can have length 64. Evaluating the position is on
    /// the hottest path of the engine, so we avoid re-scanning the whole vector
    /// on every call just to disambiguate the cache format.
    pub feat_is_pattern_ids: bool,

    /// Feature cache / pattern IDs.
    ///
    /// - When length == 64: legacy per-square abs digits (0/1/2).
    /// - When length == 64 (EGEV2): absolute ternary IDs for the 64 symmetry features.
    pub feat_id_abs: Vec<u16>,
}

#[inline(always)]
fn bit_of_sq(sq: u8) -> u64 {
    1u64 << (sq as u64)
}

impl Board {
    /// New empty board.
    pub fn new_empty(side: Color, feat_len: usize) -> Self {
        let mut b = Self {
            player: 0,
            opponent: 0,
            side,
            empty_count: 64,
            hash: 0,
            feat_is_pattern_ids: false,
            feat_id_abs: vec![0u16; feat_len],
        };
        b.hash = zobrist::compute_hash(b.bits_by_color(), b.side);
        b.recompute_feat_if_square_digits();
        b
    }

    /// Standard Othello starting position.
    pub fn new_start(feat_len: usize) -> Self {
        // Internal bitpos indices (row-major):
        //   D4=27, E4=28, D5=35, E5=36 in row-major if A1=0.
        // Sonetto uses row-major (row<<3|col) with A1=0.
        // The canonical start is:
        //   White at D4 (27) and E5 (36)
        //   Black at E4 (28) and D5 (35)
        let w1 = 27u8;
        let w2 = 36u8;
        let b1 = 28u8;
        let b2 = 35u8;

        let black = bit_of_sq(b1) | bit_of_sq(b2);
        let white = bit_of_sq(w1) | bit_of_sq(w2);

        let side = Color::Black;
        let (player, opponent) = if side == Color::Black {
            (black, white)
        } else {
            (white, black)
        };

        let mut b = Self {
            player,
            opponent,
            side,
            empty_count: 60,
            hash: 0,
            feat_is_pattern_ids: false,
            feat_id_abs: vec![0u16; feat_len],
        };
        b.hash = zobrist::compute_hash([black, white], b.side);
        b.recompute_feat_if_square_digits();
        b
    }

    /// Occupied squares bitboard.
    #[inline(always)]
    pub fn occupied(&self) -> u64 {
        self.player | self.opponent
    }

    /// Empty squares bitboard.
    #[inline(always)]
    pub fn empties(&self) -> u64 {
        !self.occupied()
    }

    /// Discs of the side-to-move.
    #[inline(always)]
    pub fn me(&self) -> u64 {
        self.player
    }

    /// Discs of the opponent.
    #[inline(always)]
    pub fn opp(&self) -> u64 {
        self.opponent
    }

    /// Absolute black bitboard.
    #[inline(always)]
    pub fn black(&self) -> u64 {
        match self.side {
            Color::Black => self.player,
            Color::White => self.opponent,
        }
    }

    /// Absolute white bitboard.
    #[inline(always)]
    pub fn white(&self) -> u64 {
        match self.side {
            Color::Black => self.opponent,
            Color::White => self.player,
        }
    }

    /// Bitboard for an absolute color.
    #[inline(always)]
    pub fn bits_of(&self, color: Color) -> u64 {
        match color {
            Color::Black => self.black(),
            Color::White => self.white(),
        }
    }

    /// `[black, white]` bitboards in absolute color order.
    #[inline(always)]
    pub fn bits_by_color(&self) -> [u64; 2] {
        [self.black(), self.white()]
    }

    // ---------------------------------------------------------------------
    // Make/Undo
    // ---------------------------------------------------------------------

    /// Apply a move (or PASS). Optional `OccMap` enables incremental pattern updates.
    #[inline(always)]
    pub fn apply_move_with_occ(&mut self, mv: Move, undo: &mut Undo, occ: Option<&OccMap>) -> bool {
        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        // Legacy per-square digits mode (feat_id_abs[0..64] are 0/1/2):
        // decide *before* mutating the bitboards.
        let do_sq_digits = occ.is_none() && self.is_square_digit_cache();

        // Core make.
        let ok = self.apply_move_core(mv, undo);
        if !ok {
            return false;
        }

        // Feature updates.
        if let Some(occ) = occ {
            // Only update incremental pattern IDs when we know they are valid.
            // (The root caller should run `recompute_features_in_place` once.)
            if self.feat_is_pattern_ids {
                update_pattern_features_for_move(self, undo, occ);
            }
        } else if do_sq_digits {
            self.apply_feat_update_for_move(undo);
        }

        true
    }

    /// Apply a move (non-PASS) using a **precomputed flips bitboard**.
    ///
    /// This is a performance helper for search paths that already computed
    /// `flips` during move ordering, avoiding a redundant flip generation
    /// inside [`Board::apply_move_with_occ`].
    ///
    /// Safety/robustness:
    /// - If `mv == PASS` or `flips == 0`, this **falls back** to
    ///   [`Board::apply_move_with_occ`].
    /// - Basic legality checks (move is empty, flips non-zero) are preserved.
    #[inline(always)]
    pub fn apply_move_with_occ_preflips(
        &mut self,
        mv: Move,
        flips: u64,
        undo: &mut Undo,
        occ: Option<&OccMap>,
    ) -> bool {
        // PASS and "no cached flips" fall back to the canonical implementation.
        if mv == PASS || flips == 0 {
            return self.apply_move_with_occ(mv, undo, occ);
        }

        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        // Legacy per-square digits mode (feat_id_abs[0..64] are 0/1/2):
        // decide *before* mutating the bitboards.
        let do_sq_digits = occ.is_none() && self.is_square_digit_cache();

        // Core make with provided flips.
        let ok = self.apply_move_core_preflips(mv, flips, undo);
        if !ok {
            return false;
        }

        // Feature updates.
        if let Some(occ) = occ {
            // Only update incremental pattern IDs when we know they are valid.
            // (The root caller should run `recompute_features_in_place` once.)
            if self.feat_is_pattern_ids {
                update_pattern_features_for_move(self, undo, occ);
            }
        } else if do_sq_digits {
            self.apply_feat_update_for_move(undo);
        }

        true
    }


    /// Apply a move (or PASS) **without updating any feature caches**.
    ///
    /// This is a performance helper for the exact endgame solver: when we only
    /// care about bitboards + hash + empty_count, maintaining incremental feature
    /// state (pattern IDs / per-square digits) is wasted work.
    #[inline(always)]
    pub fn apply_move_no_features(&mut self, mv: Move, undo: &mut Undo) -> bool {
        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        self.apply_move_core(mv, undo)
    }

    /// Apply a move (non-PASS) using a **precomputed flips bitboard**, without
    /// updating any feature caches.
    #[inline(always)]
    pub fn apply_move_no_features_preflips(&mut self, mv: Move, flips: u64, undo: &mut Undo) -> bool {
        if mv == PASS || flips == 0 {
            return self.apply_move_no_features(mv, undo);
        }

        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        self.apply_move_core_preflips(mv, flips, undo)
    }


    // ---------------------------------------------------------------------
    // P0-7: Unchecked hot-path make-move helpers (search/exact)
    // ---------------------------------------------------------------------

    /// Apply a **non-PASS** move without updating any feature caches, assuming it is legal.
    ///
    /// This is a hot-path helper for the exact solver / search where moves are
    /// generated from `legal_moves()` and therefore do not need redundant
    /// bounds/occupied/flips==0 checks.
    ///
    /// # Safety
    /// - `mv` must be in `[0,63]` and not `PASS`.
    /// - The destination square must be empty.
    /// - The move must be legal (i.e. flips are non-zero).
    #[inline(always)]
    pub unsafe fn apply_move_no_features_unchecked(&mut self, mv: Move, undo: &mut Undo) {
        debug_assert!(mv != PASS);
        debug_assert!(mv < 64);
        debug_assert!(self.empty_count > 0);

        let mv_bit = bit_of_sq(mv);
        debug_assert!((mv_bit & (self.player | self.opponent)) == 0);

        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        let me = self.player;
        let opp = self.opponent;

        let flips = flips_for_move_unchecked(me, opp, mv_bit);
        debug_assert!(flips != 0);
        debug_assert!((flips & mv_bit) == 0);
        debug_assert!((flips & me) == 0);

        // Record undo (must be done before side changes).
        undo.mv = mv;
        undo.mv_bit = mv_bit;
        undo.flips = flips;

        // Update bitboards (still from mover perspective).
        self.player = me | mv_bit | flips;
        self.opponent = opp & !flips;

        // Update empty count.
        self.empty_count -= 1;

        // Hash update (absolute color keys).
        let mover_color = self.side;
        let opp_color = mover_color.other();

        self.hash ^= zobrist::piece_key(mover_color, mv); // placed
        self.hash ^= zobrist::xor_piece_keys(opp_color, flips);
        self.hash ^= zobrist::xor_piece_keys(mover_color, flips);

        // Switch side, toggle side hash, and swap player/opponent orientation.
        self.side = opp_color;
        self.hash ^= zobrist::side_key();
        core::mem::swap(&mut self.player, &mut self.opponent);
    }

    /// Apply a **non-PASS** move with a precomputed `flips` set, without updating
    /// any feature caches, assuming it is legal.
    ///
    /// # Safety
    /// - `mv` must be in `[0,63]` and not `PASS`.
    /// - The destination square must be empty.
    /// - `flips` must be non-zero and consistent with the position.
    #[inline(always)]
    pub unsafe fn apply_move_no_features_preflips_unchecked(&mut self, mv: Move, flips: u64, undo: &mut Undo) {
        debug_assert!(mv != PASS);
        debug_assert!(mv < 64);
        debug_assert!(flips != 0);
        debug_assert!(self.empty_count > 0);

        let mv_bit = bit_of_sq(mv);
        debug_assert!((mv_bit & (self.player | self.opponent)) == 0);
        debug_assert!((flips & mv_bit) == 0);
        debug_assert!((flips & self.player) == 0);

        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        let me = self.player;
        let opp = self.opponent;

        // Record undo (must be done before side changes).
        undo.mv = mv;
        undo.mv_bit = mv_bit;
        undo.flips = flips;

        // Update bitboards (still from mover perspective).
        self.player = me | mv_bit | flips;
        self.opponent = opp & !flips;

        // Update empty count.
        self.empty_count -= 1;

        // Hash update (absolute color keys).
        let mover_color = self.side;
        let opp_color = mover_color.other();

        self.hash ^= zobrist::piece_key(mover_color, mv); // placed
        self.hash ^= zobrist::xor_piece_keys(opp_color, flips);
        self.hash ^= zobrist::xor_piece_keys(mover_color, flips);

        // Switch side, toggle side hash, and swap player/opponent orientation.
        self.side = opp_color;
        self.hash ^= zobrist::side_key();
        core::mem::swap(&mut self.player, &mut self.opponent);
    }

    /// Apply a **non-PASS** move **without updating the Zobrist hash**, assuming the move is legal.
    ///
    /// This is used by the SenseiAB search backend: native Sensei's alpha-beta uses the raw
    /// bitboards as the transposition key (not Zobrist), so maintaining Zobrist during search is
    /// pure overhead.
    ///
    /// # Safety
    /// - `mv` must be in `[0,63]` and not `PASS`.
    /// - The destination square must be empty.
    /// - `flips` must be the exact flips for this move (non-zero, disjoint from player and `mv_bit`).
    #[inline(always)]
    pub unsafe fn apply_move_no_features_preflips_unchecked_nohash(
        &mut self,
        mv: Move,
        flips: u64,
        undo: &mut Undo,
    ) {
        debug_assert!(mv != PASS);
        debug_assert!(mv < 64);

        let mv_bit = bit_of_sq(mv);
        debug_assert!((mv_bit & (self.player | self.opponent)) == 0);
        debug_assert!(flips != 0);
        debug_assert!((flips & mv_bit) == 0);
        debug_assert!((flips & self.player) == 0);

        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        // Record undo (must be done before side changes).
        undo.mv = mv;
        undo.mv_bit = mv_bit;
        undo.flips = flips;

        // Update bitboards (still from mover perspective).
        let me = self.player;
        let opp = self.opponent;
        self.player = me | mv_bit | flips;
        self.opponent = opp & !flips;
        self.empty_count -= 1;

        // Switch side and swap player/opponent orientation (no hash updates).
        self.side = self.side.other();
        core::mem::swap(&mut self.player, &mut self.opponent);
    }

    /// Apply a PASS **without updating the Zobrist hash**.
    #[inline(always)]
    pub fn apply_pass_nohash(&mut self, undo: &mut Undo) {
        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        // Record undo.
        undo.mv = PASS;
        undo.mv_bit = 0;
        undo.flips = 0;

        // Switch side and swap orientation (no hash updates).
        self.side = self.side.other();
        core::mem::swap(&mut self.player, &mut self.opponent);
    }

    /// Apply a **non-PASS** move with a precomputed `flips` set and an OccMap,
    /// assuming the move is legal.
    ///
    /// This is the preferred hot path for the recursive midgame search.
    ///
    /// # Safety
    /// - `mv` must be in `[0,63]` and not `PASS`.
    /// - The destination square must be empty.
    /// - `flips` must be non-zero and consistent with the position.
    #[inline(always)]
    pub unsafe fn apply_move_with_occ_preflips_unchecked(
        &mut self,
        mv: Move,
        flips: u64,
        undo: &mut Undo,
        occ: &OccMap,
    ) {
        debug_assert!(mv != PASS);
        debug_assert!(mv < 64);
        debug_assert!(flips != 0);
        debug_assert!(self.empty_count > 0);

        let mv_bit = bit_of_sq(mv);
        debug_assert!((mv_bit & (self.player | self.opponent)) == 0);
        debug_assert!((flips & mv_bit) == 0);
        debug_assert!((flips & self.player) == 0);

        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        let me = self.player;
        let opp = self.opponent;

        // Record undo (must be done before side changes).
        undo.mv = mv;
        undo.mv_bit = mv_bit;
        undo.flips = flips;

        // Update bitboards (still from mover perspective).
        self.player = me | mv_bit | flips;
        self.opponent = opp & !flips;

        // Update empty count.
        self.empty_count -= 1;

        // Hash update (absolute color keys).
        let mover_color = self.side;
        let opp_color = mover_color.other();

        self.hash ^= zobrist::piece_key(mover_color, mv); // placed
        self.hash ^= zobrist::xor_piece_keys(opp_color, flips);
        self.hash ^= zobrist::xor_piece_keys(mover_color, flips);

        // Switch side, toggle side hash, and swap player/opponent orientation.
        self.side = opp_color;
        self.hash ^= zobrist::side_key();
        core::mem::swap(&mut self.player, &mut self.opponent);

        // Incremental pattern feature IDs (Sensei-style) when enabled.
        if self.feat_is_pattern_ids {
            update_pattern_features_for_move(self, undo, occ);
        }
    }

    /// Apply a **non-PASS** move with a precomputed `flips` set and an OccMap **without updating the
    /// Zobrist hash**, assuming the move is legal.
    ///
    /// This keeps incremental pattern feature updates (when enabled), but skips hash maintenance.
    /// Native Sensei's alpha-beta uses the raw bitboards as the transposition key, so the wasm
    /// port treats Zobrist updates as avoidable overhead in the hot path.
    ///
    /// # Safety
    /// - `mv` must be in `[0,63]` and not `PASS`.
    /// - The destination square must be empty.
    /// - `flips` must be the exact flips for this move (non-zero, disjoint from `mv_bit` and
    ///   `self.player`).
    #[inline(always)]
    pub unsafe fn apply_move_with_occ_preflips_unchecked_nohash(
        &mut self,
        mv: Move,
        flips: u64,
        undo: &mut Undo,
        occ: &OccMap,
    ) {
        debug_assert!(mv != PASS);
        debug_assert!(mv < 64);
        debug_assert!(flips != 0);
        debug_assert!(self.empty_count > 0);

        let mv_bit = bit_of_sq(mv);
        debug_assert!((mv_bit & (self.player | self.opponent)) == 0);
        debug_assert!((flips & mv_bit) == 0);
        debug_assert!((flips & self.player) == 0);

        // Save state needed for undo.
        undo.old_hash = self.hash;
        undo.old_empty = self.empty_count;
        undo.old_side = self.side;

        let me = self.player;
        let opp = self.opponent;

        // Record undo (must be done before side changes).
        undo.mv = mv;
        undo.mv_bit = mv_bit;
        undo.flips = flips;

        // Update bitboards (still from mover perspective).
        self.player = me | mv_bit | flips;
        self.opponent = opp & !flips;

        // Update empty count.
        self.empty_count -= 1;

        // Switch side and swap player/opponent orientation (no hash updates).
        self.side = self.side.other();
        core::mem::swap(&mut self.player, &mut self.opponent);

        if self.feat_is_pattern_ids {
            update_pattern_features_for_move(self, undo, occ);
        }
    }

    /// Apply a **non-PASS** move with an OccMap, assuming the move is legal.
    ///
    /// # Safety
    /// - `mv` must be in `[0,63]` and not `PASS`.
    /// - The destination square must be empty.
    /// - The move must be legal (i.e. flips are non-zero).
    #[inline(always)]
    pub unsafe fn apply_move_with_occ_unchecked(&mut self, mv: Move, undo: &mut Undo, occ: &OccMap) {
        debug_assert!(mv != PASS);
        debug_assert!(mv < 64);

        let mv_bit = bit_of_sq(mv);
        debug_assert!((mv_bit & (self.player | self.opponent)) == 0);

        let flips = flips_for_move_unchecked(self.player, self.opponent, mv_bit);
        debug_assert!(flips != 0);

        self.apply_move_with_occ_preflips_unchecked(mv, flips, undo, occ);
    }


    /// Undo a move previously applied by [`apply_move_no_features`].
    #[inline(always)]
    pub fn undo_move_no_features(&mut self, undo: &Undo) {
        self.undo_move_core(undo);
    }

    /// Apply a move without an OccMap (legacy / tests).
    #[inline(always)]
    pub fn apply_move(&mut self, mv: Move, undo: &mut Undo) -> bool {
        self.apply_move_with_occ(mv, undo, None)
    }

    /// Undo a move previously applied by [`apply_move_with_occ`].
    #[inline(always)]
    pub fn undo_move_with_occ(&mut self, undo: &Undo, occ: Option<&OccMap>) {
        // Rollback features first (uses undo.old_side/mv/flips).
        if let Some(occ) = occ {
            if self.feat_is_pattern_ids {
                rollback_pattern_features_for_move(self, undo, occ);
            }
        } else {
            if self.is_square_digit_cache() {
                self.undo_feat_update_for_move(undo);
            }
        }

        self.undo_move_core(undo);
    }

    /// Undo a move without an OccMap (legacy / tests).
    #[inline(always)]
    pub fn undo_move(&mut self, undo: &Undo) {
        self.undo_move_with_occ(undo, None)
    }

    #[inline(always)]
    fn apply_move_core(&mut self, mv: Move, undo: &mut Undo) -> bool {
        // PASS: side changes, hash toggles side, and player/opponent swap.
        if mv == PASS {
            undo.mv = PASS;
            undo.mv_bit = 0;
            undo.flips = 0;

            self.side = self.side.other();
            self.hash ^= zobrist::side_key();
            core::mem::swap(&mut self.player, &mut self.opponent);
            return true;
        }

        if mv > 63 {
            return false;
        }

        let mv_bit = bit_of_sq(mv);

        // mv_bit must be empty.
        if (mv_bit & (self.player | self.opponent)) != 0 {
            return false;
        }

        let me = self.player;
        let opp = self.opponent;

        // Compute flips.
        // We already verified `mv_bit` is empty, so we can skip redundant checks.
        let flips = flips_for_move_unchecked(me, opp, mv_bit);
        if flips == 0 {
            return false;
        }

        // Record undo (must be done before side changes).
        undo.mv = mv;
        undo.mv_bit = mv_bit;
        undo.flips = flips;

        // Update bitboards (still from mover perspective).
        self.player = me | mv_bit | flips;
        self.opponent = opp & !flips;

        // Update empty count.
        if self.empty_count == 0 {
            // Defensive guard: a non-PASS move on a full board is illegal.
            return false;
        }
        self.empty_count -= 1;

        // Hash update (absolute color keys).
        let mover_color = self.side;
        let opp_color = mover_color.other();

        self.hash ^= zobrist::piece_key(mover_color, mv); // placed
        self.hash ^= zobrist::xor_piece_keys(opp_color, flips);
        self.hash ^= zobrist::xor_piece_keys(mover_color, flips);

        // Switch side, toggle side hash, and swap player/opponent orientation.
        self.side = opp_color;
        self.hash ^= zobrist::side_key();
        core::mem::swap(&mut self.player, &mut self.opponent);

        true
    }


    /// Core make routine using a precomputed `flips` set.
    ///
    /// This mirrors [`Board::apply_move_core`] but skips flip generation.
    #[inline(always)]
    fn apply_move_core_preflips(&mut self, mv: Move, flips: u64, undo: &mut Undo) -> bool {
        debug_assert!(mv != PASS);

        if mv > 63 {
            return false;
        }

        let mv_bit = bit_of_sq(mv);

        // mv_bit must be empty.
        if (mv_bit & (self.player | self.opponent)) != 0 {
            return false;
        }

        // flips must be non-zero for a legal move.
        if flips == 0 {
            return false;
        }

        let me = self.player;
        let opp = self.opponent;

        // Defensive sanity checks (debug-only):
        // - flips never overlap mover discs
        // - flips never include the placed square
        debug_assert!((flips & me) == 0);
        debug_assert!((flips & mv_bit) == 0);

        // Record undo (must be done before side changes).
        undo.mv = mv;
        undo.mv_bit = mv_bit;
        undo.flips = flips;

        // Update bitboards (still from mover perspective).
        self.player = me | mv_bit | flips;
        self.opponent = opp & !flips;

        // Update empty count.
        if self.empty_count == 0 {
            // Defensive guard: a non-PASS move on a full board is illegal.
            return false;
        }
        self.empty_count -= 1;

        // Hash update (absolute color keys).
        let mover_color = self.side;
        let opp_color = mover_color.other();

        self.hash ^= zobrist::piece_key(mover_color, mv); // placed
        self.hash ^= zobrist::xor_piece_keys(opp_color, flips);
        self.hash ^= zobrist::xor_piece_keys(mover_color, flips);

        // Switch side, toggle side hash, and swap player/opponent orientation.
        self.side = opp_color;
        self.hash ^= zobrist::side_key();
        core::mem::swap(&mut self.player, &mut self.opponent);

        true
    }

    #[inline(always)]
    fn undo_move_core(&mut self, undo: &Undo) {
        // Restore scalars.
        self.hash = undo.old_hash;
        self.side = undo.old_side;
        self.empty_count = undo.old_empty;

        // Restore (player,opponent) orientation back to the mover.
        //
        // After a make, Board is always oriented to the *next* player.
        // Undo must swap back first so bitboard diffs apply to the mover's `player`.
        core::mem::swap(&mut self.player, &mut self.opponent);

        if undo.mv == PASS {
            return;
        }

        // Undo: remove placed + flips from mover, give flips back to opponent.
        self.player ^= undo.mv_bit | undo.flips;
        self.opponent ^= undo.flips;
    }

    // ---------------------------------------------------------------------
    // Legacy per-square digit cache helpers
    // ---------------------------------------------------------------------

    /// Does `feat_id_abs` currently represent 64 per-square digits?
    #[inline(always)]
    fn is_square_digit_cache(&self) -> bool {
        if self.feat_is_pattern_ids {
            return false;
        }
        if self.feat_id_abs.len() != 64 {
            return false;
        }

        let black = self.black();
        let white = self.white();
        if (black & white) != 0 {
            return false;
        }

        for i in 0..64usize {
            let bit = 1u64 << (i as u64);
            let expect: u16 = if (black & bit) != 0 {
                Color::Black.digit_abs()
            } else if (white & bit) != 0 {
                Color::White.digit_abs()
            } else {
                0
            };
            if self.feat_id_abs[i] != expect {
                return false;
            }
        }
        true
    }

    /// If `feat_id_abs.len()==64`, rebuild the 0/1/2 per-square digit cache.
    fn recompute_feat_if_square_digits(&mut self) {
        if self.feat_id_abs.len() != 64 {
            return;
        }
        // This path maintains the legacy per-square digit cache (0/1/2).
        self.feat_is_pattern_ids = false;
        // 0 empty
        self.feat_id_abs.fill(0);

        // Set black squares.
        let mut b = self.black();
        while b != 0 {
            let sq = b.trailing_zeros() as usize;
            b &= b - 1;
            self.feat_id_abs[sq] = Color::Black.digit_abs();
        }
        // Set white squares.
        let mut w = self.white();
        while w != 0 {
            let sq = w.trailing_zeros() as usize;
            w &= w - 1;
            self.feat_id_abs[sq] = Color::White.digit_abs();
        }
    }

    #[inline(always)]
    fn apply_feat_update_for_move(&mut self, undo: &Undo) {
        if self.feat_id_abs.len() != 64 || undo.mv == PASS {
            return;
        }

        // Legacy digit mode.
        self.feat_is_pattern_ids = false;

        // mover is undo.old_side
        let mover = undo.old_side;
        let mover_digit = mover.digit_abs();

        // placed square: empty -> mover
        self.feat_id_abs[undo.mv as usize] = mover_digit;

        // flipped squares: opp -> mover
        let mut f = undo.flips;
        while f != 0 {
            let sq = f.trailing_zeros() as usize;
            f &= f - 1;
            self.feat_id_abs[sq] = mover_digit;
        }
    }

    #[inline(always)]
    fn undo_feat_update_for_move(&mut self, undo: &Undo) {
        if self.feat_id_abs.len() != 64 || undo.mv == PASS {
            return;
        }

        // Legacy digit mode.
        self.feat_is_pattern_ids = false;

        // mover is undo.old_side; after undo, flips revert to opponent
        let opp_digit = undo.old_side.other().digit_abs();

        // placed square: mover -> empty
        self.feat_id_abs[undo.mv as usize] = 0;

        // flipped squares: mover -> opp
        let mut f = undo.flips;
        while f != 0 {
            let sq = f.trailing_zeros() as usize;
            f &= f - 1;
            self.feat_id_abs[sq] = opp_digit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_undo_pass_changes_only_side_hash_and_orientation() {
        let mut b = Board::new_start(0);
        let before_player = b.player;
        let before_opp = b.opponent;
        let before_empty = b.empty_count;
        let before_hash = b.hash;
        let before_side = b.side;

        let mut u = Undo::default();
        assert!(b.apply_move(PASS, &mut u));

        // Pieces unchanged, but orientation swapped.
        assert_eq!(b.player, before_opp);
        assert_eq!(b.opponent, before_player);
        assert_eq!(b.empty_count, before_empty);
        assert_ne!(b.hash, before_hash);
        assert_eq!(b.side, before_side.other());

        b.undo_move(&u);
        assert_eq!(b.player, before_player);
        assert_eq!(b.opponent, before_opp);
        assert_eq!(b.empty_count, before_empty);
        assert_eq!(b.hash, before_hash);
        assert_eq!(b.side, before_side);
    }

    #[test]
    fn start_position_hash_is_deterministic() {
        let b1 = Board::new_start(0);
        let b2 = Board::new_start(0);
        assert_eq!(b1.player, b2.player);
        assert_eq!(b1.opponent, b2.opponent);
        assert_eq!(b1.side, b2.side);
        assert_eq!(b1.hash, b2.hash);
    }

    #[test]
    fn feat_square_digits_path_updates_in_place() {
        let mut b = Board::new_start(64);
        // Ensure feature digits look sane at start: 4 discs.
        let mut cnt = 0usize;
        for &d in &b.feat_id_abs {
            if d != 0 {
                cnt += 1;
            }
        }
        assert_eq!(cnt, 4);

        // Apply a known legal opening move for black from start.
        let mv = 19u8;
        let mut u = Undo::default();
        let ok = b.apply_move(mv, &mut u);
        if !ok {
            return;
        }
        assert_ne!(b.feat_id_abs[mv as usize], 0);

        b.undo_move(&u);
        let mut cnt2 = 0usize;
        for &d in &b.feat_id_abs {
            if d != 0 {
                cnt2 += 1;
            }
        }
        assert_eq!(cnt2, 4);
    }
}
