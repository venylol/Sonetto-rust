//! Sensei "extras".
//!
//! Small, self-contained ports of Sensei utilities / estimators.
//! These are used by the Rust searcher where it improves move ordering and
//! pruning behavior.

pub mod endgame_time;
pub mod move_selector;
pub mod win_probability;

// (phase 7) Next:
// pub mod error_margin;
// pub mod endgame_options;
// pub mod opening_options;
