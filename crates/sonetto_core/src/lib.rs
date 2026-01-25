//! sonetto-core: Othello/Reversi engine core.
//!
//! ## Coordinate contract (critical)
//! Engine-internal squares are **bitpos** in `[0,63]` with **row-major** layout:
//! `bitpos = (row<<3)|col`.
//!
//! External/UI often uses **ext packed**: `ext=(col<<3)|row`.
//! Conversion helpers live in `coord` and must be applied at the boundary.

pub mod apply;
pub mod backend;
pub mod board;
pub mod coord;
pub mod eval_egev2;
pub mod eval;
pub mod egev2;

pub mod features {
    pub mod occ;
    pub mod swap;
    pub mod update;
}

pub mod flips;
pub mod movegen;
pub mod score;
pub mod search;
pub mod derivative;
pub mod stability;
pub mod tt;
pub mod zobrist;

/// Optional ports of small Sensei utilities (not used by default search).
pub mod sensei_extras;

/// Sensei-style alpha-beta backend (opt-in; coexists with `search::Searcher`).
pub mod sensei_ab;
