//! Engine backend abstraction.
//!
//! Sonetto currently ships a single in-tree search implementation
//! ([`crate::search::Searcher`]).
//!
//! Phase 2 introduces an *additional, opt-in* backend:
//! - [`crate::sensei_ab::SenseiAlphaBeta`]: a fixed-depth alpha-beta search
//!   inspired by OthelloSensei's `EvaluatorAlphaBeta`, but using Sonetto's
//!   **EGEV2 evaluation** (no weight format changes).
//!
//! This module is still **not wired into the default runtime path** (e.g. WASM
//! bindings) in Phase 2. The intent is to make both backends available behind a
//! single enum so that higher layers can select at runtime, and can trivially
//! roll back to the existing `Searcher`.

use crate::board::Board;
use crate::coord::Move;
use crate::score::Score;
use crate::search::{AnalyzeTopNRequest, AnalyzeTopNResult, SearchLimits, SearchOutcome, Searcher};
use crate::sensei_ab::SenseiAlphaBeta;

/// Identifies which search backend is being used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendKind {
    /// The existing backend: `sonetto_core::search` (Sonetto/Egaroucid hybrid).
    Sonetto,
    /// Sensei-style fixed-depth alpha-beta (`sensei_ab`).
    SenseiAlphaBeta,
}

impl Default for BackendKind {
    #[inline]
    fn default() -> Self {
        BackendKind::Sonetto
    }
}

/// A concrete backend handle.
///
/// This is an enum (instead of a trait object) so WASM bindings can avoid
/// dynamic dispatch and keep code size predictable.
pub enum EngineBackend {
    /// The existing Sonetto backend.
    Sonetto(Searcher),
    /// The Sensei-style alpha-beta backend.
    SenseiAlphaBeta(SenseiAlphaBeta),
}


impl core::fmt::Debug for EngineBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EngineBackend::Sonetto(_) => f.write_str("EngineBackend::Sonetto(..)"),
            EngineBackend::SenseiAlphaBeta(_) => f.write_str("EngineBackend::SenseiAlphaBeta(..)"),
        }
    }
}

impl EngineBackend {
    /// Returns which backend variant this value holds.
    #[inline]
    pub fn kind(&self) -> BackendKind {
        match self {
            EngineBackend::Sonetto(_) => BackendKind::Sonetto,
            EngineBackend::SenseiAlphaBeta(_) => BackendKind::SenseiAlphaBeta,
        }
    }

    /// Compute a best move under the given search limits, at a fixed depth.
    ///
    /// The semantics of `depth` match `Searcher::search`: ply depth.
    #[inline]
    pub fn best_move(
        &mut self,
        board: &mut Board,
        depth: u8,
        limits: SearchLimits,
    ) -> (Move, Score) {
        match self {
            EngineBackend::Sonetto(searcher) => {
                // Keep the existing behavior unchanged.
                let SearchOutcome { best_move, score, .. } =
                    searcher.search_with_limits(board, -1_000_000, 1_000_000, depth, limits);
                (best_move, score)
            }
            EngineBackend::SenseiAlphaBeta(ab) => ab.best_move(board, depth, limits),
        }
    }

    /// Analyze and return a Top-N report.
    #[inline]
    pub fn analyze_top_n(&mut self, board: &mut Board, req: AnalyzeTopNRequest) -> AnalyzeTopNResult {
        match self {
            EngineBackend::Sonetto(searcher) => searcher.analyze_top_n(board, req),
            EngineBackend::SenseiAlphaBeta(ab) => ab.analyze_top_n(board, req),
        }
    }
}
