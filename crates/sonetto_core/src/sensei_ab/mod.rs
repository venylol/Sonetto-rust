//! Sensei-style alpha-beta backend.
//!
//! Phase 2 goal: port the *search* shape of OthelloSensei's
//! `EvaluatorAlphaBeta` while **keeping Sonetto's evaluation pipeline**
//! (EGEV2 / `egbk3` / `egev2` file formats and loaders) unchanged.
//!
//! - Leaf eval must go through [`crate::eval`] (EGEV2).
//! - Move ordering is implemented via Sensei-inspired iterators.
//! - This module is **opt-in** and coexists with the default
//!   [`crate::search::Searcher`] backend.

pub mod alpha_beta;
pub mod move_iter;
pub mod hash_map;

// Internal helpers mirroring Sensei's "depth-one" evaluator and "last moves" solver.
// These are kept private to minimize API surface and reduce risk.
pub(crate) mod depth_one;
pub(crate) mod last_moves;

pub use alpha_beta::SenseiAlphaBeta;
pub use move_iter::MoveIteratorKind;
