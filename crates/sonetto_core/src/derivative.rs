//! Derivative-style best-first tree scheduler (Stage 5).
//!
//! This module is a port of the core ideas behind OthelloSensei's
//! `evaluatederivative`.
//!
//! It contains:
//! - A single-threaded scheduler (default).
//! - An optional **multi-threaded leaf-update** variant (Sensei-native style),
//!   enabled via the `parallel_rayon` feature and the `*_mt` entrypoints.
//!
//! - Fixed-size arena (`TreeNodeSupplier` analogue) with transpositions.
//! - Strong bounds (`lower`/`upper`) and a dynamic *weak* focus window
//!   (`weak_lower`/`weak_upper`).
//! - Greedy best-first selection of the next leaf to update based on:
//!   uncertainty, goal-closeness, and novelty.
//! - Expansion with per-child `quick_eval` and an adaptive seed depth.
//! - Budgeted exact proof attempt (`SolvePosition`), falling back to expansion
//!   on budget abort.
//!
//! The code is designed to be **safe by construction**:
//! - Arena indices are validated at allocation time; iterators only traverse
//!   previously allocated link IDs.
//! - Strong bounds are updated monotonically (`lower` non-decreasing,
//!   `upper` non-increasing).
//! - The main scheduler loop either tightens bounds, expands a leaf into an
//!   internal node, or terminates due to budget/memory limits.

use crate::board::{Board, Color, Undo};
use crate::coord::{Move, PASS};
use crate::eval::{score_disc, N_PATTERN_FEATURES};
use crate::features::occ::OccMap;
use crate::features::update::recompute_features_in_place;
use crate::movegen::{legal_moves, push_moves_from_mask};
use crate::score::{disc_diff_scaled, Score, MAX_DISC_DIFF, SCALE};
use crate::search::{SearchLimits, SearchOutcome, Searcher};

use core::cmp::{max, min};

// Parallel scheduler support is optional (kept behind a feature so that
// default/Wasm builds remain dependency-light and deterministic).
#[cfg(feature = "parallel_rayon")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "parallel_rayon")]
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// Abstraction over the underlying alpha-beta engine used by [`DerivativeEvaluator`].
///
/// This allows the derivative scheduler to drive either Sonetto's [`Searcher`] or the
/// Sensei-style [`crate::sensei_ab::SenseiAlphaBeta`] backend without changing the
/// scheduler logic.
///
/// The contract is intentionally minimal: the scheduler only needs:
/// - a cheap `quick_eval` signal for ordering + heuristics
/// - budgeted fixed-depth alpha-beta (`search_with_limits`)
/// - budgeted exact-to-end proof search (`exact_search_with_limits`)
pub trait AlphaBetaBackend {
    /// Occurrence map used for incremental pattern feature updates.
    fn occ(&self) -> &OccMap;

    /// Lightweight evaluation used for child ordering and seed-depth heuristics.
    ///
    /// This should be fast, deterministic, and reasonably correlated with the true value.
    fn quick_eval(&self, board: &Board) -> Score;

    /// Budgeted fixed-depth alpha-beta search.
    fn search_with_limits(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        depth: u8,
        limits: SearchLimits,
    ) -> SearchOutcome;

    /// Budgeted exact solve-to-end search (or as exact as the backend can provide).
    ///
    /// The returned `score` must be sound for tightening the caller's `[lower, upper]`
    /// window: if the true value is outside `[alpha, beta]`, the result must be a bound.
    fn exact_search_with_limits(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        limits: SearchLimits,
    ) -> SearchOutcome;
}

impl AlphaBetaBackend for Searcher {
    #[inline(always)]
    fn occ(&self) -> &OccMap {
        &self.occ
    }

    #[inline(always)]
    fn quick_eval(&self, board: &Board) -> Score {
        // P1-4: `quick_eval` is primarily used for child ordering + seed-depth heuristics.
        // Use the raw disc score (no tanh mapping) for a lightweight, monotone signal.
        //
        // Fast path: if `feat_id_abs` contains precomputed pattern IDs, `score_disc`
        // avoids recomputing ternary indices from bitboards.
        score_disc(board, &self.weights) * SCALE
    }

    #[inline(always)]
    fn search_with_limits(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        depth: u8,
        limits: SearchLimits,
    ) -> SearchOutcome {
        Searcher::search_with_limits(self, board, alpha, beta, depth, limits)
    }

    #[inline(always)]
    fn exact_search_with_limits(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        limits: SearchLimits,
    ) -> SearchOutcome {
        Searcher::exact_search_with_limits(self, board, alpha, beta, limits)
    }
}

/// Outcome status for a derivative evaluation run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivativeStatus {
    /// The root position was proven exactly (`lower == upper`).
    Solved,
    /// The caller-provided budget was exhausted.
    Budget,
    /// The fixed arena filled up (nodes or link pools).
    ArenaFull,
    /// The position has no legal moves for either side (terminal at root).
    NoMoves,
    /// Internal safety stop to prevent infinite loops.
    MaxIterations,
}

/// Result returned by [`DerivativeEvaluator::evaluate_with_bounds`].
#[derive(Clone, Copy, Debug)]
pub struct DerivativeResult {
    pub best_move: Move,
    pub estimate: Score,
    pub lower: Score,
    pub upper: Score,
    pub weak_lower: Score,
    pub weak_upper: Score,
    pub nodes_used: u64,
    pub iterations: u32,
    pub tree_nodes: u32,
    pub status: DerivativeStatus,
}

/// Tunables for the derivative scheduler.
///
/// The defaults are chosen to be conservative (small arena, bounded seed work).
#[derive(Clone, Copy, Debug)]
pub struct DerivativeConfig {
    /// Number of worker threads for the **parallel** derivative scheduler.
    ///
    /// - `1` means single-threaded (default, Wasm-friendly).
    /// - `0` means "auto" (use Rayon thread pool size).
    ///
    /// Notes:
    /// - This knob is only used by the `*_mt` entrypoints (enabled by the
    ///   `sonetto_core/parallel_rayon` feature).
    /// - The sequential `evaluate_*` APIs ignore it.
    pub num_threads: usize,

    /// Maximum number of tree nodes.
    pub max_tree_nodes: usize,
    /// Hash table size used for transpositions. Rounded up to a power of two.
    pub hash_size: usize,
    /// Maximum number of child edges (sum over all expanded nodes).
    pub max_child_links: usize,
    /// Maximum number of father edges (used by transpositions).
    pub max_father_links: usize,

    /// Minimum seed depth in `AddChildren`.
    pub seed_depth_min: u8,
    /// Maximum seed depth in `AddChildren`.
    pub seed_depth_max: u8,
    /// Node budget for a single seed search call.
    pub seed_max_nodes: u64,

    /// Minimum node budget for a single exact proof attempt.
    pub solve_min_nodes: u64,
    /// Maximum node budget for a single exact proof attempt.
    pub solve_max_nodes: u64,
    /// Do not attempt proof when empties exceed this threshold.
    pub solve_max_empties: u8,

    /// If the *strong* bound width is below this threshold, proof is preferred.
    pub solve_prefer_width: Score,

    /// Minimum half-width of the global weak window.
    pub weak_min_half_width: Score,

    /// Maximum number of scheduler iterations.
    pub max_iterations: u32,

    /// Selection weight: uncertainty (bigger => more urgent).
    pub w_uncertainty: f64,
    /// Selection weight: goal distance (smaller distance => more urgent).
    pub w_goal_dist: f64,
    /// Selection weight: novelty (less updated => more urgent).
    pub w_novelty: f64,

    /// Selection weight: penalize nodes currently being worked on by other threads.
    ///
    /// This is a lightweight analogue of Sensei's `n_threads_working_` avoidance.
    /// It has **no effect** in the single-threaded scheduler.
    pub w_threads_working: f64,
}

impl Default for DerivativeConfig {
    fn default() -> Self {
        // Keep defaults modest for Wasm; native callers can increase if needed.
        let max_tree_nodes = 120_000usize;
        let max_child_links = 2_000_000usize;
        let max_father_links = 300_000usize;

        Self {
            // Default to 1 for deterministic, Wasm-friendly behavior.
            num_threads: 1,
            max_tree_nodes,
            hash_size: 1 << 19, // 524,288
            max_child_links,
            max_father_links,

            seed_depth_min: 2,
            seed_depth_max: 5,
            seed_max_nodes: 80_000,

            solve_min_nodes: 50_000,
            solve_max_nodes: 2_000_000,
            solve_max_empties: 30,

            // About 2 discs in SCALE=32.
            solve_prefer_width: 2 * SCALE,

            // About 4 discs in SCALE=32.
            weak_min_half_width: 4 * SCALE,

            max_iterations: 50_000,

            w_uncertainty: 1.0,
            w_goal_dist: 0.25,
            w_novelty: 0.5,

            // Mild avoidance by default (only affects parallel path).
            w_threads_working: 0.25,
        }
    }
}

impl DerivativeConfig {
    /// Return a copy of this config with arena sizes adjusted for a target `tree_node_cap`.
    ///
    /// # Sensei mapping
    ///
    /// This corresponds to choosing the capacity of Sensei's `TreeNodeSupplier` arena.
    /// Stage 6 surfaces this as `tree_node_cap` so the unified `analyze_top_n` entrypoint
    /// can budget memory use when `AnalyzeTopNStrategy::Derivative` is selected.
    ///
    /// # What gets scaled
    ///
    /// The derivative scheduler maintains several fixed-size arenas:
    /// - `max_tree_nodes`: number of tree nodes
    /// - `max_child_links`: adjacency list edges (`ChildLink`)
    /// - `max_father_links`: reverse edges for transposition merging (`FatherLink`)
    /// - `hash_size`: hash index backing the `TreeNodeSupplier` (rounded to power-of-two)
    ///
    /// This helper keeps the default link-per-node ratios roughly intact:
    /// - child links: ~17 per node
    /// - father links: ~3 per node
    pub fn with_tree_node_cap(mut self, tree_node_cap: usize) -> Self {
        if tree_node_cap == 0 {
            return self;
        }

        self.max_tree_nodes = tree_node_cap;

        // Preserve the default ratios (120k nodes -> 2,000,000 child links -> ~16.7/link).
        self.max_child_links = tree_node_cap
            .saturating_mul(17)
            .max(1024);

        // Father links scale lower; typical ratio is a small constant factor.
        self.max_father_links = tree_node_cap
            .saturating_mul(3)
            .max(1024);

        // Keep the index roughly 4x the node cap (like default 524,288 vs 120,000).
        let target_hash = tree_node_cap.saturating_mul(4).max(1024);
        self.hash_size = target_hash;

        self
    }
}


/// Main entry type.
///
/// Reuse a single instance to amortize the arena allocation cost.
pub struct DerivativeEvaluator {
    cfg: DerivativeConfig,
    arena: TreeNodeSupplier,

    // Root state for the current run.
    root: NodeId,
    root_lower: Score,
    root_upper: Score,
    weak_lower: Score,
    weak_upper: Score,

    total_nodes_used: u64,
    iterations: u32,

    // P0-11: scratch buffer for pattern feature IDs (avoid per-expand allocation)
    scratch_feat: Vec<u16>,
}

impl DerivativeEvaluator {
    pub fn new(cfg: DerivativeConfig) -> Self {
        Self {
            cfg,
            arena: TreeNodeSupplier::new(cfg),
            root: 0,
            root_lower: score_min(),
            root_upper: score_max(),
            weak_lower: score_min(),
            weak_upper: score_max(),
            total_nodes_used: 0,
            iterations: 0,

            // P0-11: reuse this buffer when expanding nodes.
            scratch_feat: Vec::with_capacity(N_PATTERN_FEATURES),
        }
    }

    /// Reconfigure this evaluator for a new run.
    ///
    /// If the requested arena sizes fit within the currently allocated buffers,
    /// we reuse the existing allocations; otherwise we rebuild the evaluator.
    ///
    /// This is important for Wasm/UI usage: it avoids large per-call heap
    /// allocations when derivative analysis is invoked repeatedly with the same
    /// (or smaller) `tree_node_cap`.
    pub fn reconfigure(&mut self, cfg: DerivativeConfig) {
        let want_hash = cfg.hash_size.max(1024).next_power_of_two();

        let can_reuse = self.arena.nodes.len() >= cfg.max_tree_nodes
            && self.arena.child_links.len() >= cfg.max_child_links.saturating_add(1)
            && self.arena.father_links.len() >= cfg.max_father_links.saturating_add(1)
            && self.arena.index.len() >= want_hash;

        if can_reuse {
            self.cfg = cfg;
            self.arena.cfg = cfg;
        } else {
            *self = DerivativeEvaluator::new(cfg);
        }
    }

    /// Evaluate using a selectable alpha-beta backend in the full legal score range.
    #[inline]
    pub fn evaluate_backend<B: AlphaBetaBackend>(
        &mut self,
        backend: &mut B,
        root_board: &Board,
        limits: SearchLimits,
    ) -> DerivativeResult {
        self.evaluate_with_bounds_backend(backend, root_board, score_min(), score_max(), limits)
    }

    /// Convenience wrapper: keep the original API that takes Sonetto's [`Searcher`].
    #[inline]
    pub fn evaluate(&mut self, searcher: &mut Searcher, root_board: &Board, limits: SearchLimits) -> DerivativeResult {
        self.evaluate_backend(searcher, root_board, limits)
    }

    /// Run derivative best-first scheduling starting from `root_board`.
    ///
    /// - `lower`/`upper` define the initial strong bounds for the root.
    /// - `limits.max_nodes` is treated as the total *cross-call* node budget.
    pub fn evaluate_with_bounds_backend<B: AlphaBetaBackend>(
        &mut self,
        backend: &mut B,
        root_board: &Board,
        mut lower: Score,
        mut upper: Score,
        limits: SearchLimits,
    ) -> DerivativeResult {
        // Normalize bounds.
        if lower > upper {
            core::mem::swap(&mut lower, &mut upper);
        }
        lower = lower.clamp(score_min(), score_max());
        upper = upper.clamp(score_min(), score_max());
        if lower > upper {
            // Degenerate input: clamp produced inversion.
            lower = score_min();
            upper = score_max();
        }

        self.arena.reset();
        self.total_nodes_used = 0;
        self.iterations = 0;

        self.root_lower = lower;
        self.root_upper = upper;
        self.weak_lower = lower;
        self.weak_upper = upper;

        if self.root_lower > self.root_upper {
            core::mem::swap(&mut self.root_lower, &mut self.root_upper);
            core::mem::swap(&mut self.weak_lower, &mut self.weak_upper);
        }

        // Create root node.
        let root_hash = root_board.hash;
        let root_bits = [root_board.player, root_board.opponent];
        let root_side = root_board.side;
        let root_empty = root_board.empty_count;

        let (root_id, _is_new) = match self.arena.get_or_create(root_hash, 0, root_bits, root_side, root_empty) {
            Ok(v) => v,
            Err(_) => {
                return DerivativeResult {
                    best_move: PASS,
                    estimate: 0,
                    lower,
                    upper,
                    weak_lower: lower,
                    weak_upper: upper,
                    nodes_used: 0,
                    iterations: 0,
                    tree_nodes: 0,
                    status: DerivativeStatus::ArenaFull,
                };
            }
        };
        self.root = root_id;

        // Initialize strong + weak bounds.
        {
            let n = self.arena.node_mut(root_id);
            n.lower = lower;
            n.upper = upper;
            n.weak_lower = lower;
            n.weak_upper = upper;
            n.depth = 0;

            // Quick eval at root.
            let b = root_board.clone();
            n.leaf_eval = backend.quick_eval(&b);
            n.est = n.leaf_eval.clamp(n.lower, n.upper);
            n.eval_depth = 0;
            n.has_eval = true;
        }

        // Expand root once to build an initial frontier.
        let budget_total = limits.max_nodes.unwrap_or(u64::MAX);
        let mut status = DerivativeStatus::Budget;

        let used0 = match self.expand_leaf(backend, root_id, self.root_eval_goal(), budget_total) {
            Ok(u) => u,
            Err(ArenaError::Full) => {
                status = DerivativeStatus::ArenaFull;
                0
            }
        };
        self.total_nodes_used = self.total_nodes_used.saturating_add(used0);
        self.arena.propagate_descendants(root_id, used0);

        // Terminal-at-root special case.
        if self.arena.node(root_id).is_terminal && self.arena.node(root_id).is_solved() {
            status = DerivativeStatus::NoMoves;
        }

        // Main loop.
        while self.iterations < self.cfg.max_iterations {
            self.iterations += 1;

            // Stop: budget exhausted.
            if self.total_nodes_used >= budget_total {
                status = DerivativeStatus::Budget;
                break;
            }

            // Stop: solved.
            if self.arena.node(root_id).is_solved() {
                status = DerivativeStatus::Solved;
                break;
            }

            // Maintain global weak window.
            self.update_global_weak_bounds();

            // Pick a leaf to update.
            let leaf_sel = match self.best_descendant() {
                Some(s) => s,
                None => {
                    // No selectable leaf (should be rare). Treat as budget stop.
                    status = DerivativeStatus::Budget;
                    break;
                }
            };

            // Compute remaining budget for this step.
            let budget_left = budget_total - self.total_nodes_used;

            let step_used = if self.to_be_solved(leaf_sel.leaf) {
                match self.solve_position(backend, leaf_sel.leaf, leaf_sel.eval_goal, leaf_sel.alpha, leaf_sel.beta, budget_left) {
                    Ok(u) => u,
                    Err(ArenaError::Full) => {
                        status = DerivativeStatus::ArenaFull;
                        break;
                    }
                }
            } else {
                match self.expand_leaf(backend, leaf_sel.leaf, leaf_sel.eval_goal, budget_left) {
                    Ok(u) => u,
                    Err(ArenaError::Full) => {
                        status = DerivativeStatus::ArenaFull;
                        break;
                    }
                }
            };

            self.total_nodes_used = self.total_nodes_used.saturating_add(step_used);
            self.arena.propagate_descendants(leaf_sel.leaf, step_used);

            // After any change, recompute fathers (DAG-safe).
            self.arena.update_fathers_from_child(leaf_sel.leaf);
        }

        if self.iterations >= self.cfg.max_iterations {
            status = DerivativeStatus::MaxIterations;
        }

        let (best_move, estimate) = self.best_move_and_estimate();
        let root = self.arena.node(root_id);

        DerivativeResult {
            best_move,
            estimate,
            lower: root.lower,
            upper: root.upper,
            weak_lower: self.weak_lower,
            weak_upper: self.weak_upper,
            nodes_used: self.total_nodes_used,
            iterations: self.iterations,
            tree_nodes: self.arena.num_nodes(),
            status,
        }
    }

    /// Backwards-compatible wrapper: evaluate using Sonetto's [`Searcher`].
    #[inline]
    pub fn evaluate_with_bounds(
        &mut self,
        searcher: &mut Searcher,
        root_board: &Board,
        lower: Score,
        upper: Score,
        limits: SearchLimits,
    ) -> DerivativeResult {
        self.evaluate_with_bounds_backend(searcher, root_board, lower, upper, limits)
    }

    // ---------------------------------------------------------------------
    // Parallel entrypoints (Rayon)
    // ---------------------------------------------------------------------

    /// Parallel derivative evaluation using a **per-thread** alpha-beta backend.
    ///
    /// This mirrors Sensei native's multi-threaded derivative scheduler:
    /// - threads share a single derivative tree
    /// - each thread repeatedly selects a leaf (best-first) and updates it
    /// - leaf updates are protected by a per-leaf lock
    /// - the global weak window is maintained via a cheap atomic guard
    ///
    /// The parallel implementation is gated behind the `sonetto_core/parallel_rayon`
    /// feature so that default builds remain small and deterministic.
    #[cfg(feature = "parallel_rayon")]
    pub fn evaluate_backend_mt<B, F>(
        &mut self,
        make_backend: F,
        root_board: &Board,
        limits: SearchLimits,
    ) -> DerivativeResult
    where
        B: AlphaBetaBackend + Send + 'static,
        F: Fn(usize, usize) -> B + Send + Sync,
    {
        self.evaluate_with_bounds_backend_mt(make_backend, root_board, score_min(), score_max(), limits)
    }

    /// Parallel derivative evaluation with an explicit initial bound window.
    #[cfg(feature = "parallel_rayon")]
    pub fn evaluate_with_bounds_backend_mt<B, F>(
        &mut self,
        make_backend: F,
        root_board: &Board,
        lower: Score,
        upper: Score,
        limits: SearchLimits,
    ) -> DerivativeResult
    where
        B: AlphaBetaBackend + Send + 'static,
        F: Fn(usize, usize) -> B + Send + Sync,
    {
        parallel::evaluate_derivative_mt(self, make_backend, root_board, lower, upper, limits)
    }

    /// Convenience helper: parallel derivative evaluation using cloned `Searcher` workers.
    ///
    /// This keeps total TT memory roughly constant by splitting the caller's TT
    /// budget across workers (Sensei-style "root split" memory budgeting).
    #[cfg(feature = "parallel_rayon")]
    pub fn evaluate_mt_from_searcher(&mut self, searcher: &Searcher, root_board: &Board, limits: SearchLimits) -> DerivativeResult {
        let tt_mb = searcher.tt_mb();
        let weights = searcher.weights.clone();
        let feats = searcher.feats.clone();
        let swap = searcher.swap.clone();
        let occ = searcher.occ.clone();

        self.evaluate_backend_mt(
            move |_tid, n_threads| {
                let worker_tt_mb: usize = (tt_mb / n_threads.max(1)).max(1);
                Searcher::new(worker_tt_mb, weights.clone(), feats.clone(), swap.clone(), occ.clone())
            },
            root_board,
            limits,
        )
    }


    // -------------------------------------------------------------------------
    // Weak bounds
    // -------------------------------------------------------------------------

    /// Root eval goal: clamp the current root estimate into the global weak window.
    #[inline(always)]
    fn root_eval_goal(&self) -> Score {
        let root = self.arena.node(self.root);
        root.est.clamp(self.weak_lower, self.weak_upper)
    }

    /// Update the global weak window based on the root's current strong bounds
    /// and point estimate.
    fn update_global_weak_bounds(&mut self) {
        let root = self.arena.node(self.root);
        let strong_lo = root.lower;
        let strong_hi = root.upper;

        // If the root is already tight, mirror it.
        if strong_lo >= strong_hi {
            self.weak_lower = strong_lo;
            self.weak_upper = strong_hi;
            return;
        }

        let center = root.est.clamp(strong_lo, strong_hi);
        let width = (strong_hi - strong_lo).abs();

        // Shrink to 1/4 of the strong uncertainty, but never below the minimum.
        let mut half = max(self.cfg.weak_min_half_width, width / 4);
        // Never exceed the strong interval.
        half = min(half, (strong_hi - strong_lo) / 2);

        let mut new_lo = center.saturating_sub(half);
        let mut new_hi = center.saturating_add(half);
        new_lo = new_lo.max(strong_lo);
        new_hi = new_hi.min(strong_hi);

        if new_lo >= new_hi {
            // Fall back to the strong interval if the computed window collapsed.
            new_lo = strong_lo;
            new_hi = strong_hi;
        }

        // Sensei-style extension rule: when the expected window moves outside
        // the current one, extend only one side.
        if new_lo < self.weak_lower {
            self.weak_lower = new_lo;
        } else if new_hi > self.weak_upper {
            self.weak_upper = new_hi;
        } else {
            self.weak_lower = new_lo;
            self.weak_upper = new_hi;
        }

        // Keep the root node in sync.
        {
            let n = self.arena.node_mut(self.root);
            n.weak_lower = self.weak_lower.max(n.lower);
            n.weak_upper = self.weak_upper.min(n.upper);
            if n.weak_lower >= n.weak_upper {
                n.weak_lower = n.lower;
                n.weak_upper = n.upper;
            }
        }
    }

    /// Compute the weak window for a node at depth `d`.
    ///
    /// In negamax, child values are negated. Therefore, the root weak window
    /// `[L, U]` maps to `[-U, -L]` at odd depths.
    #[inline(always)]
    fn weak_for_depth(&self, d: u8) -> (Score, Score) {
        if (d & 1) == 0 {
            (self.weak_lower, self.weak_upper)
        } else {
            (-self.weak_upper, -self.weak_lower)
        }
    }

    // -------------------------------------------------------------------------
    // Best-first selection
    // -------------------------------------------------------------------------

    fn best_descendant(&mut self) -> Option<LeafSelection> {
        let mut node_id = self.root;

        let mut eval_goal = self.root_eval_goal();
        let mut alpha = self.root_lower;
        let mut beta = self.root_upper;

        // Walk down greedily.
        loop {
            let depth = self.arena.node(node_id).depth;
            let (wlo, whi) = self.weak_for_depth(depth);

            // Update the node-local weak bounds (Sensei-style extension rule).
            {
                let n = self.arena.node_mut(node_id);

                let mut target_lo = max(n.lower, wlo);
                let mut target_hi = min(n.upper, whi);
                if target_lo >= target_hi {
                    target_lo = n.lower;
                    target_hi = n.upper;
                }

                if target_lo < n.weak_lower {
                    n.weak_lower = target_lo;
                } else if target_hi > n.weak_upper {
                    n.weak_upper = target_hi;
                } else {
                    n.weak_lower = target_lo;
                    n.weak_upper = target_hi;
                }
            }

            // Clamp the working alpha/beta window.
            {
                let n = self.arena.node(node_id);
                alpha = max(alpha, n.lower);
                beta = min(beta, n.upper);
                alpha = max(alpha, n.weak_lower);
                beta = min(beta, n.weak_upper);
                if alpha >= beta {
                    // The node is effectively solved within the current focus.
                    // Return this node as the leaf (it will be treated as solved).
                    let a = min(alpha, beta);
                    let b = max(alpha, beta);
                    return Some(LeafSelection {
                        leaf: node_id,
                        eval_goal: eval_goal.clamp(a, b),
                        alpha: a,
                        beta: b,
                    });
                }
            }

            eval_goal = eval_goal.clamp(alpha, beta);

            // If leaf, return.
            if self.arena.node(node_id).is_leaf {
                return Some(LeafSelection {
                    leaf: node_id,
                    eval_goal,
                    alpha,
                    beta,
                });
            }

            // Pick best child.
            let best_child = self.best_child(node_id, eval_goal)?;

            // Move to child: negamax window transform.
            let tmp_alpha = alpha;
            alpha = -beta;
            beta = -tmp_alpha;
            eval_goal = -eval_goal;

            node_id = best_child;
        }
    }

    fn best_child(&mut self, node_id: NodeId, goal: Score) -> Option<NodeId> {
        let mut link = self.arena.node(node_id).child_head;
        if link == 0 {
            return None;
        }

        let mut best: Option<NodeId> = None;
        let mut best_score: f64 = f64::NEG_INFINITY;

        while link != 0 {
            let e = self.arena.child_link(link);
            let child_id = e.child;

            // Update child's weak window lazily based on the current global window.
            {
                let depth = self.arena.node(child_id).depth;
                let (wlo, whi) = self.weak_for_depth(depth);
                let c = self.arena.node_mut(child_id);
                let mut tlo = max(c.lower, wlo);
                let mut thi = min(c.upper, whi);
                if tlo >= thi {
                    tlo = c.lower;
                    thi = c.upper;
                }

                if tlo < c.weak_lower {
                    c.weak_lower = tlo;
                } else if thi > c.weak_upper {
                    c.weak_upper = thi;
                } else {
                    c.weak_lower = tlo;
                    c.weak_upper = thi;
                }
            }

            let child = self.arena.node(child_id);

            // Parent perspective: negate child's estimate.
            let move_est = -child.est;

            // Uncertainty proxy: width of child's weak interval.
            let weak_w = (child.weak_upper - child.weak_lower).abs() as f64 / (SCALE as f64);

            // Goal-closeness: smaller distance is better.
            let dist = (move_est - goal).abs() as f64 / (SCALE as f64);

            // Novelty: prefer nodes that have been updated fewer times.
            let novelty = 1.0 / (1.0 + child.n_updates as f64);

            let score = self.cfg.w_uncertainty * weak_w
                - self.cfg.w_goal_dist * dist
                + self.cfg.w_novelty * novelty;

            if score > best_score {
                best_score = score;
                best = Some(child_id);
            }

            link = e.next;
        }

        best
    }

    // -------------------------------------------------------------------------
    // Expansion + seeding
    // -------------------------------------------------------------------------

    fn expand_leaf<B: AlphaBetaBackend>(
        &mut self,
        backend: &mut B,
        node_id: NodeId,
        eval_goal: Score,
        budget_left: u64,
    ) -> Result<u64, ArenaError> {
        if budget_left == 0 {
            return Ok(0);
        }

        // Only expand if still a leaf.
        if !self.arena.node(node_id).is_leaf {
            return Ok(0);
        }

        let mut b = self.arena.node(node_id).to_board();

        // -----------------------------------------------------------------
        // P1-4: lightweight depth-one evaluator setup
        // -----------------------------------------------------------------
        //
        // Sensei keeps per-thread pattern state ("EvaluatorDepthOne") so that
        // per-child `quick_eval` is cheap: it computes pattern ids once for the
        // parent and then updates/undoes them incrementally while iterating
        // moves.
        //
        // Sonetto's derivative arena intentionally stores *only* the minimal
        // position state (bitboards/hash/side/empties) to keep memory bounded.
        // We therefore attach a scratch `feat_id_abs` buffer **only while
        // expanding** this node:
        // - Recompute absolute pattern IDs once from bitboards.
        // - Use `apply_move_with_occ` / `undo_move_with_occ` so feature ids are
        //   updated incrementally for each move.
        //
        // This mirrors the main search hot path and avoids the slow
        // bitboard->ternary feature decode for every `quick_eval` call.
        // P0-11: avoid allocating a fresh feature buffer per expansion.
        // We reuse a per-evaluator scratch Vec and return it on scope exit.
        struct FeatScratchGuard {
            scratch: *mut Vec<u16>,
            board_feat: *mut Vec<u16>,
        }

        impl Drop for FeatScratchGuard {
            fn drop(&mut self) {
                // Safety: pointers are valid for the guard's lifetime.
                unsafe {
                    let mut buf = core::mem::take(&mut *self.board_feat);
                    buf.clear();
                    *self.scratch = buf;
                }
            }
        }

        // Attach scratch feature buffer iff we have an OccMap.
        // (Without Occ, the main search uses disc-diff, so we skip pattern work.)
        let has_occ = !backend.occ().is_empty();
        let _feat_guard = if has_occ {
            // Move scratch -> board, recompute once, then keep it updated incrementally.
            let mut buf = core::mem::take(&mut self.scratch_feat);
            buf.clear();
            buf.resize(N_PATTERN_FEATURES, 0);
            b.feat_id_abs = buf;
            recompute_features_in_place(&mut b, backend.occ());

            // Guard returns the buffer to `self.scratch_feat` even on early returns.
            Some(FeatScratchGuard {
                scratch: &mut self.scratch_feat as *mut Vec<u16>,
                board_feat: &mut b.feat_id_abs as *mut Vec<u16>,
            })
        } else {
            None
        };


        // Generate moves.
        let me = b.player;
        let op = b.opponent;

        let moves_mask = legal_moves(me, op);
        if moves_mask == 0 {
            let opp_mask = legal_moves(op, me);
            if opp_mask == 0 {
                // Terminal.
                let v = disc_diff_scaled(&b, b.side);
                {
                    let n = self.arena.node_mut(node_id);
                    n.lower = v;
                    n.upper = v;
                    n.leaf_eval = v;
                    n.est = v;
                    n.is_terminal = true;
                    n.is_leaf = true;
                }
                return Ok(1);
            }
        }

        // Enumerate legal moves (or PASS).
        let mut moves: [Move; 64] = [PASS; 64];
        let n_moves: usize = if moves_mask != 0 {
            push_moves_from_mask(moves_mask, &mut moves)
        } else {
            // Pass only.
            moves[0] = PASS;
            1
        };

        // Mark as internal before adding children to prevent re-entrancy.
        {
            let n = self.arena.node_mut(node_id);
            n.is_leaf = false;
            n.child_head = 0;
            n.child_count = 0;
        }

        let parent_depth = self.arena.node(node_id).depth;
        let parent_work = remaining_work_estimate(b.empty_count, n_moves as u8);

        let mut used: u64 = 1;

        // Apply each move, create child node, attach edges, and seed eval.
        let mut undo = Undo::default();
        for i in 0..n_moves {
            let mv = moves[i];
            let ok = if has_occ {
                // Maintain pattern IDs in the scratch buffer.
                b.apply_move_with_occ(mv, &mut undo, Some(backend.occ()))
            } else {
                b.apply_move(mv, &mut undo)
            };
            if !ok {
                continue;
            }

            // Ensure hash is correct (Board maintains it incrementally).
            let child_hash = b.hash;
            let child_bits = [b.player, b.opponent];
            let child_side = b.side;
            let child_empty = b.empty_count;
            let child_depth = parent_depth.saturating_add(1);

            let (child_id, is_new) = self.arena.get_or_create(child_hash, child_depth, child_bits, child_side, child_empty)?;

            // Parent -> child edge.
            self.arena.add_child(node_id, mv, child_id)?;
            // Child remembers father (DAG update propagation).
            self.arena.add_father(child_id, node_id)?;

            // If new, initialize bounds and quick eval.
            if is_new {
                let q = backend.quick_eval(&b);

                let (wlo, whi) = self.weak_for_depth(child_depth);
                let weak_min = self.cfg.weak_min_half_width;
                {
                    let c = self.arena.node_mut(child_id);
                    c.lower = score_min();
                    c.upper = score_max();
                    c.depth = child_depth;
                    c.leaf_eval = q;
                    c.est = q;
                    c.eval_depth = 0;
                    c.has_eval = true;

                    // Initialize local weak window around `q`, but constrained by the
                    // global weak window mapped to this depth.
                    let radius = max(weak_min, (c.upper - c.lower) / 8);
                    let local_lo = q.saturating_sub(radius);
                    let local_hi = q.saturating_add(radius);
                    c.weak_lower = max(c.lower, max(wlo, local_lo));
                    c.weak_upper = min(c.upper, min(whi, local_hi));
                    if c.weak_lower >= c.weak_upper {
                        c.weak_lower = max(c.lower, wlo);
                        c.weak_upper = min(c.upper, whi);
                        if c.weak_lower >= c.weak_upper {
                            c.weak_lower = c.lower;
                            c.weak_upper = c.upper;
                        }
                    }
                }

                // Seed depth adaptively.
                let child_goal = -eval_goal;
                let seed_depth = choose_seed_depth(self.cfg, parent_depth, parent_work, (q - child_goal).abs());

                // Respect remaining budget.
                let seed_budget = min(self.cfg.seed_max_nodes, budget_left.saturating_sub(used));

                if seed_budget > 0 {
                    // Seed search uses midgame search at shallow depth.
                    let out = backend.search_with_limits(
                        &mut b,
                        score_min(),
                        score_max(),
                        seed_depth,
                        SearchLimits {
                            max_nodes: Some(seed_budget),
                        },
                    );
                    used = used.saturating_add(out.nodes);

                    // Update only if the search produced a better-informed value.
                    if !out.aborted {
                        let c = self.arena.node_mut(child_id);
                        c.leaf_eval = out.score;
                        c.est = out.score.clamp(c.lower, c.upper);
                        c.eval_depth = seed_depth;
                    }
                }
            }

            // Undo move.
            if has_occ {
                b.undo_move_with_occ(&undo, Some(backend.occ()));
            } else {
                b.undo_move(&undo);
            }

            if used >= budget_left {
                break;
            }
        }

        // Recompute parent estimate from children.
        self.arena.update_node_from_children(node_id);

        Ok(used.min(budget_left))
    }

    // -------------------------------------------------------------------------
    // SolvePosition
    // -------------------------------------------------------------------------

    fn solve_position<B: AlphaBetaBackend>(
        &mut self,
        backend: &mut B,
        leaf_id: NodeId,
        eval_goal: Score,
        alpha: Score,
        beta: Score,
        budget_left: u64,
    ) -> Result<u64, ArenaError> {
        if budget_left == 0 {
            return Ok(0);
        }

        let leaf = self.arena.node(leaf_id);
        if !leaf.is_leaf {
            return Ok(0);
        }

        // If terminal, nothing to do.
        if leaf.is_terminal {
            return Ok(0);
        }

        // Build board.
        let mut b = leaf.to_board();

        // Proof budget: clamp between min and max, and within remaining budget.
        let mut proof_budget = max(self.cfg.solve_min_nodes, budget_left.min(self.cfg.solve_max_nodes));
        proof_budget = min(proof_budget, budget_left);

        // Window: intersect with the leaf's current strong bounds.
        let alpha2 = max(alpha, leaf.lower);
        let beta2 = min(beta, leaf.upper);
        let (alpha2, beta2) = if alpha2 < beta2 { (alpha2, beta2) } else { (leaf.lower, leaf.upper) };

        let out = backend.exact_search_with_limits(&mut b, alpha2, beta2, SearchLimits { max_nodes: Some(proof_budget) });
        let mut used = out.nodes;

        if out.aborted {
            // Budget abort: fall back to expansion.
            let budget_after = budget_left.saturating_sub(used);
            let extra = self.expand_leaf(backend, leaf_id, eval_goal, budget_after)?;
            used = used.saturating_add(extra);
            return Ok(used.min(budget_left));
        }

        // Update bounds based on alpha/beta result classification.
        {
            let n = self.arena.node_mut(leaf_id);
            // Tighten strong bounds monotonically.
            if out.score <= alpha2 {
                n.upper = min(n.upper, out.score);
            } else if out.score >= beta2 {
                n.lower = max(n.lower, out.score);
            } else {
                n.lower = max(n.lower, out.score);
                n.upper = min(n.upper, out.score);
            }

            // Update point estimate.
            n.leaf_eval = out.score;
            n.est = out.score.clamp(n.lower, n.upper);

            if n.lower >= n.upper {
                n.lower = out.score;
                n.upper = out.score;
            }

            n.n_updates = n.n_updates.saturating_add(1);
        }

        Ok(used.min(budget_left))
    }

    // -------------------------------------------------------------------------
    // ToBeSolved
    // -------------------------------------------------------------------------

    fn to_be_solved(&self, leaf_id: NodeId) -> bool {
        let leaf = self.arena.node(leaf_id);
        if !leaf.is_leaf {
            return false;
        }
        if leaf.is_terminal || leaf.lower >= leaf.upper {
            return false;
        }
        if leaf.empty_count > self.cfg.solve_max_empties {
            return false;
        }

        let width = (leaf.upper - leaf.lower).abs();
        if width <= self.cfg.solve_prefer_width {
            return true;
        }

        // Heuristic: if the estimated full solve work is small, attempt proof.
        // Use a fresh move count estimate for leaves, because cached counts are
        // only filled for internal nodes.
        let me = leaf.bits[0];
        let opp = leaf.bits[1];
        let moves = legal_moves(me, opp);
        let mc: u8 = if moves != 0 {
            moves.count_ones() as u8
        } else {
            let opp_moves = legal_moves(opp, me);
            if opp_moves != 0 { 1 } else { 0 }
        }.max(1);

        let work = remaining_work_estimate(leaf.empty_count, mc);
        work <= (self.cfg.solve_max_nodes as f64) * 2.0
    }

    // -------------------------------------------------------------------------
    // Root reporting
    // -------------------------------------------------------------------------

    fn best_move_and_estimate(&self) -> (Move, Score) {
        let root = self.arena.node(self.root);
        if root.is_leaf {
            return (PASS, root.est);
        }

        let mut best_mv: Move = PASS;
        let mut best_val: Score = score_min();

        let mut link = root.child_head;
        while link != 0 {
            let e = self.arena.child_link(link);
            let child = self.arena.node(e.child);
            let v = -child.est;
            if v > best_val {
                best_val = v;
                best_mv = e.mv;
            }
            link = e.next;
        }

        (best_mv, best_val)
    }

    /// Return the current root move estimates as a sorted Top-N list.
    ///
    /// The returned scores are from the root perspective (i.e. the *original*
    /// side-to-move in `evaluate`). Values are the current *point estimates*
    /// maintained by the derivative scheduler.
    ///
    /// # Sensei mapping
    ///
    /// This mirrors how Sensei's `EvaluatorDerivative` reports per-root-move
    /// estimates/bounds while the scheduler is still running (or once it
    /// converges).
    ///
    /// Notes:
    /// - If the root has no legal moves, this returns an empty list (matching
    ///   the semantics of `Searcher::analyze_top_n_*`).
    /// - This does **not** force additional search; it only reports the current
    ///   state of the derivative tree.
    pub fn root_top_n_estimates(&self, top_n: usize) -> Vec<(Move, Score)> {
        let top_n = top_n.max(1);

        let root = self.arena.node(self.root);
        if root.is_leaf || root.child_head == 0 {
            return Vec::new();
        }

        let mut out: Vec<(Move, Score)> = Vec::new();
        let mut link = root.child_head;
        while link != 0 {
            let e = self.arena.child_link(link);
            let child = self.arena.node(e.child);
            // Negamax flip: root perspective = -child.
            let est = (-child.est).clamp(score_min(), score_max());
            out.push((e.mv, est));
            link = e.next;
        }

        out.sort_by(|a, b| b.1.cmp(&a.1));
        if out.len() > top_n {
            out.truncate(top_n);
        }
        out
    }

    /// Like [`DerivativeEvaluator::root_top_n_estimates`], but also reports the
    /// current *strong* bounds for each move.
    ///
    /// Returns tuples `(mv, estimate, lower, upper)`, all from the root
    /// perspective. Bounds are clamped into the legal score range.
    pub fn root_top_n_bounds(&self, top_n: usize) -> Vec<(Move, Score, Score, Score)> {
        let top_n = top_n.max(1);

        let root = self.arena.node(self.root);
        if root.is_leaf || root.child_head == 0 {
            return Vec::new();
        }

        let mut out: Vec<(Move, Score, Score, Score)> = Vec::new();
        let mut link = root.child_head;
        while link != 0 {
            let e = self.arena.child_link(link);
            let child = self.arena.node(e.child);

            // Negamax flip for point estimate.
            let est = (-child.est).clamp(score_min(), score_max());

            // Strong bounds flip + swap:
            // child in [lo, hi] (child perspective) => root in [-hi, -lo].
            let lo = (-child.upper).clamp(score_min(), score_max());
            let hi = (-child.lower).clamp(score_min(), score_max());

            out.push((e.mv, est, lo, hi));
            link = e.next;
        }

        out.sort_by(|a, b| b.1.cmp(&a.1));
        if out.len() > top_n {
            out.truncate(top_n);
        }
        out
    }
}

// -----------------------------------------------------------------------------
// Internal: arena + node graph
// -----------------------------------------------------------------------------

type NodeId = u32;

type LinkId = u32;

#[derive(Clone, Copy, Debug, Default)]
struct ChildLink {
    mv: Move,
    child: NodeId,
    next: LinkId,
}

#[derive(Clone, Copy, Debug, Default)]
struct FatherLink {
    father: NodeId,
    next: LinkId,
}

#[derive(Clone, Copy, Debug)]
struct TreeNode {
    bits: [u64; 2],
    side: Color,
    hash: u64,
    empty_count: u8,

    depth: u8,

    // Strong bounds.
    lower: Score,
    upper: Score,

    // Weak focus bounds.
    weak_lower: Score,
    weak_upper: Score,

    // Point estimate.
    leaf_eval: Score,
    est: Score,
    eval_depth: u8,
    has_eval: bool,

    // Graph connectivity.
    child_head: LinkId,
    child_count: u8,
    father_head: LinkId,
    father_count: u16,

    // Scheduler metadata.
    n_updates: u32,
    descendants: u64,
    mark: u32,

    // Cached properties.
    cached_move_count: u8,
    is_leaf: bool,
    is_terminal: bool,
}

impl Default for TreeNode {
    fn default() -> Self {
        Self {
            bits: [0u64, 0u64],
            side: Color::Black,
            hash: 0,
            empty_count: 64,
            depth: 0,

            lower: score_min(),
            upper: score_max(),
            weak_lower: score_min(),
            weak_upper: score_max(),

            leaf_eval: 0,
            est: 0,
            eval_depth: 0,
            has_eval: false,

            child_head: 0,
            child_count: 0,
            father_head: 0,
            father_count: 0,

            n_updates: 0,
            descendants: 0,
            mark: 0,

            cached_move_count: 0,
            is_leaf: true,
            is_terminal: false,
        }
    }
}

impl TreeNode {
    #[inline(always)]
    fn is_solved(&self) -> bool {
        self.lower >= self.upper
    }

    fn to_board(&self) -> Board {
        // Use stored hash (Board uses the same deterministic scheme).
        Board {
            player: self.bits[0],
            opponent: self.bits[1],
            side: self.side,
            empty_count: self.empty_count,
            hash: self.hash,
            feat_is_pattern_ids: false,
            feat_id_abs: Vec::new(),
        }
    }
}

#[derive(Debug)]
enum ArenaError {
    Full,
}

/// Fixed-size pool + transposition map.
///
/// This mirrors Sensei's `TreeNodeSupplier` concept:
/// - `nodes` is a fixed pool.
/// - `index` is an open-addressed hash table mapping (hash, depth) -> NodeId.
/// - `first_valid` implements cheap resets without clearing the full table.
struct TreeNodeSupplier {
    cfg: DerivativeConfig,

    nodes: Vec<TreeNode>,
    num_nodes: u32,

    child_links: Vec<ChildLink>,
    child_next: u32,

    father_links: Vec<FatherLink>,
    father_next: u32,

    index: Vec<u32>,
    index_mask: usize,
    first_valid: u32,

    mark_gen: u32,

    // P5-1: scratch stack reused by DAG propagation (avoid per-update Vec alloc).
    stack: Vec<NodeId>,

    // ---------------------------------------------------------------------
    // Parallel scheduler metadata (Sensei-style leaf locking).
    //
    // These are kept **separate** from `TreeNode` so that `TreeNode` can remain
    // `Copy` (cheap snapshots for worker threads).
    // ---------------------------------------------------------------------
    #[cfg(feature = "parallel_rayon")]
    leaf_lock: Vec<AtomicU32>,

    #[cfg(feature = "parallel_rayon")]
    threads_working: Vec<AtomicU32>,
}

impl TreeNodeSupplier {
    fn new(cfg: DerivativeConfig) -> Self {
        let hash_size = cfg.hash_size.max(1024).next_power_of_two();

        Self {
            cfg,
            nodes: vec![TreeNode::default(); cfg.max_tree_nodes],
            num_nodes: 0,

            // Link ID 0 is reserved as "null".
            child_links: vec![ChildLink::default(); cfg.max_child_links + 1],
            child_next: 1,

            father_links: vec![FatherLink::default(); cfg.max_father_links + 1],
            father_next: 1,

            index: vec![0u32; hash_size],
            index_mask: hash_size - 1,
            first_valid: 1,

            mark_gen: 1,

            // P5-1: scratch stack reused by DAG propagation (avoid per-update Vec alloc).
            stack: Vec::with_capacity(cfg.max_tree_nodes),

            #[cfg(feature = "parallel_rayon")]
            leaf_lock: (0..cfg.max_tree_nodes)
                .map(|_| AtomicU32::new(0))
                .collect(),

            #[cfg(feature = "parallel_rayon")]
            threads_working: (0..cfg.max_tree_nodes)
                .map(|_| AtomicU32::new(0))
                .collect(),
        }
    }

    /// Create a tiny placeholder supplier.
    ///
    /// This is used by the parallel derivative entrypoints to temporarily
    /// move the large preallocated arena out of `DerivativeEvaluator` and into
    /// a shared `Arc<Mutex<...>>` without forcing a reallocation.
    #[cfg(feature = "parallel_rayon")]
    fn placeholder(cfg: DerivativeConfig) -> Self {
        Self {
            cfg,
            nodes: Vec::new(),
            num_nodes: 0,

            child_links: Vec::new(),
            child_next: 1,

            father_links: Vec::new(),
            father_next: 1,

            // A 1-slot index is enough because the placeholder is never queried.
            index: vec![0u32; 1],
            index_mask: 0,
            first_valid: 1,

            mark_gen: 1,
            stack: Vec::new(),

            leaf_lock: Vec::new(),
            threads_working: Vec::new(),
        }
    }

    fn reset(&mut self) {
        // Cheap reset: invalidate by shifting the valid range.
        self.first_valid = self.first_valid.wrapping_add(self.num_nodes);

        // If we are close to overflow, hard reset.
        let max_nodes_u32 = self.cfg.max_tree_nodes as u32;
        if self.first_valid >= u32::MAX - max_nodes_u32 - 1 {
            for v in self.index.iter_mut() {
                *v = 0;
            }
            self.first_valid = 1;
        }

        self.num_nodes = 0;
        self.child_next = 1;
        self.father_next = 1;
        self.mark_gen = self.mark_gen.wrapping_add(1);
        if self.mark_gen == 0 {
            self.mark_gen = 1;
        }

        self.stack.clear();

        // Note: we do not clear the node/link arrays; they are overwritten on demand.
    }

    #[inline(always)]
    fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    #[inline(always)]
    fn node(&self, id: NodeId) -> &TreeNode {
        &self.nodes[id as usize]
    }

    #[inline(always)]
    fn node_mut(&mut self, id: NodeId) -> &mut TreeNode {
        &mut self.nodes[id as usize]
    }

    // ---------------------------------------------------------------------
    // Parallel scheduler metadata helpers
    // ---------------------------------------------------------------------

    /// Try to acquire the per-leaf lock.
    ///
    /// Returns `true` if this thread successfully locked the leaf.
    #[cfg(feature = "parallel_rayon")]
    #[inline(always)]
    fn try_lock_leaf(&self, id: NodeId) -> bool {
        self.leaf_lock[id as usize]
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    /// Release the per-leaf lock.
    #[cfg(feature = "parallel_rayon")]
    #[inline(always)]
    fn unlock_leaf(&self, id: NodeId) {
        self.leaf_lock[id as usize].store(0, Ordering::Release);
    }

    /// Increment `threads_working` for a node.
    #[cfg(feature = "parallel_rayon")]
    #[inline(always)]
    fn inc_threads_working(&self, id: NodeId) {
        self.threads_working[id as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement `threads_working` for a node.
    #[cfg(feature = "parallel_rayon")]
    #[inline(always)]
    fn dec_threads_working(&self, id: NodeId) {
        self.threads_working[id as usize].fetch_sub(1, Ordering::Relaxed);
    }

    /// Read `threads_working` for a node.
    #[cfg(feature = "parallel_rayon")]
    #[inline(always)]
    fn threads_working(&self, id: NodeId) -> u32 {
        self.threads_working[id as usize].load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn child_link(&self, id: LinkId) -> ChildLink {
        self.child_links[id as usize]
    }

    #[inline(always)]
    fn father_link(&self, id: LinkId) -> FatherLink {
        self.father_links[id as usize]
    }

    fn is_valid_abs(&self, abs: u32) -> bool {
        if abs == 0 {
            return false;
        }
        let fv = self.first_valid as u64;
        let n = self.num_nodes as u64;
        let v = abs as u64;
        v >= fv && v < fv + n
    }

    #[inline(always)]
    fn key_to_index(&self, hash: u64, depth: u8) -> usize {
        // Mix hash and depth.
        let h = hash ^ ((depth as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        (h as usize) & self.index_mask
    }

    fn get_or_create(
        &mut self,
        hash: u64,
        depth: u8,
        bits: [u64; 2],
        side: Color,
        empty_count: u8,
    ) -> Result<(NodeId, bool), ArenaError> {
        // Probe.
        let mut idx = self.key_to_index(hash, depth);
        for _ in 0..self.index.len() {
            let abs = self.index[idx];
            if !self.is_valid_abs(abs) {
                // Empty slot (or stale). Insert.
                let node_id = self.alloc_node(hash, depth, bits, side, empty_count)?;
                let abs_new = self.first_valid.wrapping_add(node_id);
                self.index[idx] = abs_new;
                return Ok((node_id, true));
            }

            let node_id = abs.wrapping_sub(self.first_valid);
            let n = self.node(node_id);
            if n.hash == hash && n.depth == depth && n.side == side && n.bits == bits {
                return Ok((node_id, false));
            }

            idx = (idx + 1) & self.index_mask;
        }

        Err(ArenaError::Full)
    }

    fn alloc_node(
        &mut self,
        hash: u64,
        depth: u8,
        bits: [u64; 2],
        side: Color,
        empty_count: u8,
    ) -> Result<NodeId, ArenaError> {
        if (self.num_nodes as usize) >= self.cfg.max_tree_nodes {
            return Err(ArenaError::Full);
        }
        let id = self.num_nodes;
        self.num_nodes += 1;

        let mut n = TreeNode::default();
        n.hash = hash;
        n.depth = depth;
        n.bits = bits;
        n.side = side;
        n.empty_count = empty_count;
        n.child_head = 0;
        n.child_count = 0;
        n.father_head = 0;
        n.father_count = 0;
        n.is_leaf = true;
        n.is_terminal = false;
        n.cached_move_count = 0;
        n.n_updates = 0;
        n.descendants = 0;
        n.mark = 0;
        n.lower = score_min();
        n.upper = score_max();
        n.weak_lower = score_min();
        n.weak_upper = score_max();
        n.leaf_eval = 0;
        n.est = 0;
        n.eval_depth = 0;
        n.has_eval = false;

        self.nodes[id as usize] = n;

        // Reset parallel metadata for this slot.
        #[cfg(feature = "parallel_rayon")]
        {
            // These arrays are preallocated to `max_tree_nodes`.
            self.leaf_lock[id as usize].store(0, Ordering::Relaxed);
            self.threads_working[id as usize].store(0, Ordering::Relaxed);
        }

        Ok(id)
    }

    fn alloc_child_link(&mut self, link: ChildLink) -> Result<LinkId, ArenaError> {
        if (self.child_next as usize) >= self.child_links.len() {
            return Err(ArenaError::Full);
        }
        let id = self.child_next;
        self.child_next += 1;
        self.child_links[id as usize] = link;
        Ok(id)
    }

    fn alloc_father_link(&mut self, link: FatherLink) -> Result<LinkId, ArenaError> {
        if (self.father_next as usize) >= self.father_links.len() {
            return Err(ArenaError::Full);
        }
        let id = self.father_next;
        self.father_next += 1;
        self.father_links[id as usize] = link;
        Ok(id)
    }

    fn child_has_father(&self, child: NodeId, father: NodeId) -> bool {
        let mut link = self.node(child).father_head;
        while link != 0 {
            let e = self.father_link(link);
            if e.father == father {
                return true;
            }
            link = e.next;
        }
        false
    }

    fn add_child(&mut self, parent: NodeId, mv: Move, child: NodeId) -> Result<(), ArenaError> {
        let head = self.node(parent).child_head;
        let id = self.alloc_child_link(ChildLink { mv, child, next: head })?;
        {
            let p = self.node_mut(parent);
            p.child_head = id;
            p.child_count = p.child_count.saturating_add(1);
        }
        Ok(())
    }

    fn add_father(&mut self, child: NodeId, father: NodeId) -> Result<(), ArenaError> {
        if self.child_has_father(child, father) {
            return Ok(());
        }

        let head = self.node(child).father_head;
        let id = self.alloc_father_link(FatherLink { father, next: head })?;
        {
            let c = self.node_mut(child);
            c.father_head = id;
            c.father_count = c.father_count.saturating_add(1);
        }
        Ok(())
    }

    /// Recompute `lower/upper/est` for an internal node from its children.
    fn update_node_from_children(&mut self, node_id: NodeId) {
        let child_head = self.node(node_id).child_head;
        if child_head == 0 {
            return;
        }

        let mut new_lower = score_min();
        let mut new_upper = score_min();
        let mut new_est = score_min();
        let mut move_count: u8 = 0;

        let mut link = child_head;
        while link != 0 {
            let e = self.child_link(link);
            let c = self.node(e.child);

            // Parent = max_i (-child)
            new_lower = max(new_lower, -c.upper);
            new_upper = max(new_upper, -c.lower);
            new_est = max(new_est, -c.est);

            move_count = move_count.saturating_add(1);

            link = e.next;
        }

        let n = self.node_mut(node_id);
        n.lower = max(n.lower, new_lower);
        n.upper = min(n.upper, new_upper);
        n.est = new_est.clamp(n.lower, n.upper);
        n.cached_move_count = move_count;

        if n.lower >= n.upper {
            n.lower = n.est;
            n.upper = n.est;
        }

        n.n_updates = n.n_updates.saturating_add(1);
    }

    /// DAG-safe father update after a child node changed.
    fn update_fathers_from_child(&mut self, child_id: NodeId) {
        self.mark_gen = self.mark_gen.wrapping_add(1);
        if self.mark_gen == 0 {
            self.mark_gen = 1;
        }

        let mark_gen = self.mark_gen;

        // Stack of fathers to update (reuse scratch to avoid per-call heap allocation).
        let mut stack = core::mem::take(&mut self.stack);
        stack.clear();

        let mut link = self.node(child_id).father_head;
        while link != 0 {
            let e = self.father_link(link);
            stack.push(e.father);
            link = e.next;
        }

        while let Some(f) = stack.pop() {
            let mark = self.node(f).mark;
            if mark == mark_gen {
                continue;
            }
            self.node_mut(f).mark = mark_gen;

            self.update_node_from_children(f);

            // Enqueue this father's fathers.
            let mut fl = self.node(f).father_head;
            while fl != 0 {
                let e = self.father_link(fl);
                stack.push(e.father);
                fl = e.next;
            }
        }

        stack.clear();
        self.stack = stack;
    }

    /// Add `delta` descendants to `node_id` and all its fathers (deduplicated).
    fn propagate_descendants(&mut self, node_id: NodeId, delta: u64) {
        if delta == 0 {
            return;
        }

        self.mark_gen = self.mark_gen.wrapping_add(1);
        if self.mark_gen == 0 {
            self.mark_gen = 1;
        }

        let mark_gen = self.mark_gen;

        let mut stack = core::mem::take(&mut self.stack);
        stack.clear();
        stack.push(node_id);

        while let Some(nid) = stack.pop() {
            if self.node(nid).mark == mark_gen {
                continue;
            }
            {
                let n = self.node_mut(nid);
                n.mark = mark_gen;
                n.descendants = n.descendants.saturating_add(delta);
            }

            let mut link = self.node(nid).father_head;
            while link != 0 {
                let e = self.father_link(link);
                stack.push(e.father);
                link = e.next;
            }
        }

        stack.clear();
        self.stack = stack;
    }
}

// -----------------------------------------------------------------------------
// Leaf selection bundle
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct LeafSelection {
    leaf: NodeId,
    eval_goal: Score,
    alpha: Score,
    beta: Score,
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

#[inline(always)]
fn score_min() -> Score {
    -(MAX_DISC_DIFF as Score) * SCALE
}

#[inline(always)]
fn score_max() -> Score {
    (MAX_DISC_DIFF as Score) * SCALE
}


/// Very rough estimate of remaining search work.
///
/// This is used only for adaptive seeding / proof heuristics. It is intentionally
/// capped to avoid inf/nan.
fn remaining_work_estimate(empties: u8, move_count: u8) -> f64 {
    let e = empties as f64;
    let b = (move_count.max(1) as f64).min(14.0);
    let log10 = e * b.log10();
    let log10 = log10.min(12.0); // cap at 1e12
    10f64.powf(log10)
}

fn choose_seed_depth(cfg: DerivativeConfig, parent_depth: u8, parent_work: f64, delta: Score) -> u8 {
    // Map delta to disc units for intuition.
    let delta_disc = (delta.abs() as f64) / (SCALE as f64);

    // Mirror Sensei's coarse thresholds:
    // - deeper seeding when remaining work is huge
    // - and when the child is close to the goal.
    let mut depth = if parent_depth > 0 && parent_depth <= 2 && parent_work > 100_000_000.0 {
        5
    } else if parent_work > 20_000_000.0
        || (delta_disc < 16.0 && parent_work > 10_000_000.0)
        || (delta_disc < 8.0 && parent_work > 2_000_000.0)
    {
        4
    } else if parent_work > 4_000_000.0 || (delta_disc < 16.0 && parent_work > 2_000_000.0) || (delta_disc < 8.0) {
        3
    } else {
        2
    };

    depth = depth.max(cfg.seed_depth_min).min(cfg.seed_depth_max);
    depth
}

// =============================================================================
// Parallel derivative scheduler (Sensei-style leaf updates)
// =============================================================================

#[cfg(feature = "parallel_rayon")]
mod parallel {
    use super::*;

    use rayon::prelude::*;
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};

    #[derive(Clone)]
    struct LeafSelectionMt {
        leaf: NodeId,
        eval_goal: Score,
        alpha: Score,
        beta: Score,
        /// Path from root to leaf (inclusive). Used to maintain `threads_working`.
        path: Vec<NodeId>,
        /// Snapshot of the leaf node at selection time (TreeNode is Copy).
        leaf_snapshot: TreeNode,
    }

    enum UpdatePlan {
        Solve {
            out: SearchOutcome,
            alpha2: Score,
            beta2: Score,
            used: u64,
        },
        Expand {
            plan: ExpandPlan,
            used: u64,
        },
    }

    struct ExpandPlan {
        /// If `Some`, the node is terminal and the contained value is the exact
        /// terminal eval from the node's perspective.
        terminal_value: Option<Score>,
        /// Child expansions (empty when terminal).
        children: Vec<ChildPlan>,
    }

    struct ChildPlan {
        mv: Move,
        hash: u64,
        bits: [u64; 2],
        side: Color,
        empty_count: u8,
        depth: u8,
        quick_eval: Score,
        seed: Option<(Score, u8)>,
    }

    struct SharedState {
        cfg: DerivativeConfig,
        root: NodeId,
        root_lower: Score,
        root_upper: Score,
        budget_total: u64,

        arena: Mutex<TreeNodeSupplier>,

        weak_lower: AtomicI32,
        weak_upper: AtomicI32,
        is_updating_weak: AtomicBool,

        nodes_used: AtomicU64,
        iterations: AtomicU32,

        arena_full: AtomicBool,
    }

    #[inline(always)]
    fn weak_for_depth_global(weak_lower: Score, weak_upper: Score, d: u8) -> (Score, Score) {
        if (d & 1) == 0 {
            (weak_lower, weak_upper)
        } else {
            (-weak_upper, -weak_lower)
        }
    }

    impl SharedState {
        fn budget_left(&self) -> u64 {
            self.budget_total.saturating_sub(self.nodes_used.load(Ordering::Relaxed))
        }

        fn should_stop_fast(&self) -> bool {
            if self.arena_full.load(Ordering::Relaxed) {
                return true;
            }
            if self.nodes_used.load(Ordering::Relaxed) >= self.budget_total {
                return true;
            }
            if self.iterations.load(Ordering::Relaxed) >= self.cfg.max_iterations {
                return true;
            }
            false
        }

        /// Sensei-style weak window update: protected by a cheap atomic flag.
        fn maybe_update_global_weak_bounds(&self) {
            if self
                .is_updating_weak
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                return;
            }

            {
                let mut arena = self.arena.lock().unwrap();
                let root = arena.node(self.root);
                let strong_lo = root.lower;
                let strong_hi = root.upper;

                // If already tight, mirror it.
                if strong_lo >= strong_hi {
                    self.weak_lower.store(strong_lo, Ordering::Relaxed);
                    self.weak_upper.store(strong_hi, Ordering::Relaxed);
                    let n = arena.node_mut(self.root);
                    n.weak_lower = strong_lo;
                    n.weak_upper = strong_hi;
                } else {
                    let center = root.est.clamp(strong_lo, strong_hi);
                    let width = (strong_hi - strong_lo).abs();

                    // Shrink to 1/4 of strong uncertainty, but never below min.
                    let mut half = max(self.cfg.weak_min_half_width, width / 4);
                    half = min(half, (strong_hi - strong_lo) / 2);

                    let mut new_lo = center.saturating_sub(half).max(strong_lo);
                    let mut new_hi = center.saturating_add(half).min(strong_hi);
                    if new_lo >= new_hi {
                        new_lo = strong_lo;
                        new_hi = strong_hi;
                    }

                    let cur_lo = self.weak_lower.load(Ordering::Relaxed);
                    let cur_hi = self.weak_upper.load(Ordering::Relaxed);

                    // Extension rule: extend only one side when moving outside.
                    let (wlo, whi) = if new_lo < cur_lo {
                        (new_lo, cur_hi)
                    } else if new_hi > cur_hi {
                        (cur_lo, new_hi)
                    } else {
                        (new_lo, new_hi)
                    };

                    self.weak_lower.store(wlo, Ordering::Relaxed);
                    self.weak_upper.store(whi, Ordering::Relaxed);

                    let n = arena.node_mut(self.root);
                    n.weak_lower = wlo.max(n.lower);
                    n.weak_upper = whi.min(n.upper);
                    if n.weak_lower >= n.weak_upper {
                        n.weak_lower = n.lower;
                        n.weak_upper = n.upper;
                    }
                }
            }

            self.is_updating_weak.store(false, Ordering::Release);
        }

        /// Select a leaf in a best-first manner, and lock it.
        ///
        /// Returns `None` if leaf-lock contention was hit (caller should retry)
        /// or if the tree is in an unexpected state (no selectable leaf).
        fn try_select_leaf(&self) -> Option<LeafSelectionMt> {
            let weak_lower = self.weak_lower.load(Ordering::Relaxed);
            let weak_upper = self.weak_upper.load(Ordering::Relaxed);

            let mut arena = self.arena.lock().unwrap();

            // Stop if solved.
            if arena.node(self.root).is_solved() {
                return None;
            }

            let mut node_id = self.root;

            let mut eval_goal = arena
                .node(node_id)
                .est
                .clamp(weak_lower, weak_upper);
            let mut alpha = self.root_lower;
            let mut beta = self.root_upper;

            let mut path: Vec<NodeId> = Vec::new();

            loop {
                path.push(node_id);

                let depth = arena.node(node_id).depth;
                let (wlo, whi) = weak_for_depth_global(weak_lower, weak_upper, depth);

                // Update node-local weak bounds (extension rule).
                {
                    let n = arena.node_mut(node_id);
                    let mut target_lo = max(n.lower, wlo);
                    let mut target_hi = min(n.upper, whi);
                    if target_lo >= target_hi {
                        target_lo = n.lower;
                        target_hi = n.upper;
                    }

                    if target_lo < n.weak_lower {
                        n.weak_lower = target_lo;
                    } else if target_hi > n.weak_upper {
                        n.weak_upper = target_hi;
                    } else {
                        n.weak_lower = target_lo;
                        n.weak_upper = target_hi;
                    }
                }

                // Clamp alpha/beta.
                {
                    let n = arena.node(node_id);
                    alpha = max(alpha, n.lower);
                    beta = min(beta, n.upper);
                    alpha = max(alpha, n.weak_lower);
                    beta = min(beta, n.weak_upper);
                    if alpha >= beta {
                        // Focus window collapsed. Treat this node as the leaf.
                        let a = min(alpha, beta);
                        let b = max(alpha, beta);
                        eval_goal = eval_goal.clamp(a, b);

                        // Only meaningful if it's a leaf.
                        if arena.node(node_id).is_leaf {
                            if !arena.try_lock_leaf(node_id) {
                                return None;
                            }
                            for &p in &path {
                                arena.inc_threads_working(p);
                            }
                            let snap = *arena.node(node_id);
                            return Some(LeafSelectionMt {
                                leaf: node_id,
                                eval_goal,
                                alpha: a,
                                beta: b,
                                path,
                                leaf_snapshot: snap,
                            });
                        }

                        return None;
                    }
                }

                eval_goal = eval_goal.clamp(alpha, beta);

                // Leaf?
                if arena.node(node_id).is_leaf {
                    if !arena.try_lock_leaf(node_id) {
                        return None;
                    }
                    for &p in &path {
                        arena.inc_threads_working(p);
                    }
                    let snap = *arena.node(node_id);
                    return Some(LeafSelectionMt {
                        leaf: node_id,
                        eval_goal,
                        alpha,
                        beta,
                        path,
                        leaf_snapshot: snap,
                    });
                }

                // Best child.
                let best_child = best_child_locked(&mut arena, node_id, eval_goal, weak_lower, weak_upper, self.cfg)?;

                // Negamax window transform.
                let tmp_alpha = alpha;
                alpha = -beta;
                beta = -tmp_alpha;
                eval_goal = -eval_goal;
                node_id = best_child;
            }
        }
    }

    /// Pick the best child of `node_id` for the current `goal`.
    ///
    /// This is a parallel-aware variant of `DerivativeEvaluator::best_child`.
    fn best_child_locked(
        arena: &mut TreeNodeSupplier,
        node_id: NodeId,
        goal: Score,
        weak_lower: Score,
        weak_upper: Score,
        cfg: DerivativeConfig,
    ) -> Option<NodeId> {
        let mut link = arena.node(node_id).child_head;
        if link == 0 {
            return None;
        }

        let mut best: Option<NodeId> = None;
        let mut best_score: f64 = f64::NEG_INFINITY;

        while link != 0 {
            // Split borrow: grab `child` + `next` from the edge first.
            let (child_id, next) = {
                let e = arena.child_link(link);
                (e.child, e.next)
            };

            // Update child's weak window lazily (same logic as single-thread).
            {
                let depth = arena.node(child_id).depth;
                let (wlo, whi) = weak_for_depth_global(weak_lower, weak_upper, depth);
                let c = arena.node_mut(child_id);

                let mut tlo = max(c.lower, wlo);
                let mut thi = min(c.upper, whi);
                if tlo >= thi {
                    tlo = c.lower;
                    thi = c.upper;
                }

                if tlo < c.weak_lower {
                    c.weak_lower = tlo;
                } else if thi > c.weak_upper {
                    c.weak_upper = thi;
                } else {
                    c.weak_lower = tlo;
                    c.weak_upper = thi;
                }
            }

            let child = arena.node(child_id);
            let move_est = -child.est;

            let weak_w = (child.weak_upper - child.weak_lower).abs() as f64 / (SCALE as f64);
            let dist = (move_est - goal).abs() as f64 / (SCALE as f64);
            let novelty = 1.0 / (1.0 + child.n_updates as f64);
            let tw = arena.threads_working(child_id) as f64;

            let score = cfg.w_uncertainty * weak_w
                - cfg.w_goal_dist * dist
                + cfg.w_novelty * novelty
                - cfg.w_threads_working * tw;

            if score > best_score {
                best_score = score;
                best = Some(child_id);
            }

            link = next;
        }

        best
    }

    #[inline(always)]
    fn finalize_leaf_lock(arena: &mut TreeNodeSupplier, sel: &LeafSelectionMt) {
        for &p in &sel.path {
            arena.dec_threads_working(p);
        }
        arena.unlock_leaf(sel.leaf);
    }

    fn to_be_solved_snapshot(cfg: DerivativeConfig, leaf: &TreeNode) -> bool {
        if !leaf.is_leaf {
            return false;
        }
        if leaf.is_terminal || leaf.lower >= leaf.upper {
            return false;
        }
        if leaf.empty_count > cfg.solve_max_empties {
            return false;
        }

        let width = (leaf.upper - leaf.lower).abs();
        if width <= cfg.solve_prefer_width {
            return true;
        }

        let me = leaf.bits[0];
        let opp = leaf.bits[1];
        let moves = legal_moves(me, opp);
        let mc: u8 = if moves != 0 {
            moves.count_ones() as u8
        } else {
            let opp_moves = legal_moves(opp, me);
            if opp_moves != 0 { 1 } else { 0 }
        }
        .max(1);

        let work = remaining_work_estimate(leaf.empty_count, mc);
        work <= (cfg.solve_max_nodes as f64) * 2.0
    }

    fn plan_expand<B: AlphaBetaBackend>(
        backend: &mut B,
        cfg: DerivativeConfig,
        leaf: &TreeNode,
        eval_goal: Score,
        mut budget_left: u64,
        scratch_feat: &mut Vec<u16>,
    ) -> (ExpandPlan, u64) {
        if budget_left == 0 {
            return (
                ExpandPlan {
                    terminal_value: None,
                    children: Vec::new(),
                },
                0,
            );
        }

        // We count at least 1 node for the expansion itself (Sensei-ish).
        let mut used: u64 = 1;
        budget_left = budget_left.max(1);

        let mut b = leaf.to_board();
        let has_occ = !backend.occ().is_empty();

        // Attach scratch pattern ids if we have an occ map.
        if has_occ {
            scratch_feat.clear();
            scratch_feat.resize(N_PATTERN_FEATURES, 0);
            core::mem::swap(&mut b.feat_id_abs, scratch_feat);
            recompute_features_from_bitboards(&mut b, backend.occ());
        }

        let me = b.player;
        let opp = b.opponent;
        let moves_mask = legal_moves(me, opp);

        let mut moves_arr: [Move; 64] = [PASS; 64];
        let n_moves: usize = if moves_mask != 0 {
            push_moves_from_mask(moves_mask, &mut moves_arr)
        } else {
            let opp_moves = legal_moves(opp, me);
            if opp_moves == 0 {
                // Terminal: no moves for either side.
                let v = disc_diff_scaled(&b, b.side);

                if has_occ {
                    core::mem::swap(&mut b.feat_id_abs, scratch_feat);
                    scratch_feat.clear();
                }

                return (
                    ExpandPlan {
                        terminal_value: Some(v),
                        children: Vec::new(),
                    },
                    used.min(budget_left),
                );
            }
            moves_arr[0] = PASS;
            1
        };

        let parent_depth = leaf.depth;
        let parent_work = remaining_work_estimate(b.empty_count, n_moves as u8);
        let child_goal = -eval_goal;

        let mut children: Vec<ChildPlan> = Vec::with_capacity(n_moves);
        let mut undo = Undo::default();

        for i in 0..n_moves {
            if used >= budget_left {
                break;
            }

            let mv = moves_arr[i];

            // Apply move.
            if has_occ {
                b.apply_move_with_occ(mv, &mut undo, Some(backend.occ()));
            } else {
                b.apply_move_no_features(mv, &mut undo);
            }

            let child_hash = b.hash;
            let child_bits = [b.player, b.opponent];
            let child_side = b.side;
            let child_empty = b.empty_count;
            let child_depth = parent_depth.saturating_add(1);

            // Quick eval.
            let q = backend.quick_eval(&b);

            let seed_depth = choose_seed_depth(cfg, parent_depth, parent_work, (q - child_goal).abs());

            let seed_budget = min(cfg.seed_max_nodes, budget_left.saturating_sub(used));
            let mut seed: Option<(Score, u8)> = None;
            if seed_budget > 0 {
                let out = backend.search_with_limits(
                    &mut b,
                    score_min(),
                    score_max(),
                    seed_depth,
                    SearchLimits {
                        max_nodes: Some(seed_budget),
                    },
                );
                used = used.saturating_add(out.nodes);
                if !out.aborted {
                    seed = Some((out.score, seed_depth));
                }
            }

            children.push(ChildPlan {
                mv,
                hash: child_hash,
                bits: child_bits,
                side: child_side,
                empty_count: child_empty,
                depth: child_depth,
                quick_eval: q,
                seed,
            });

            // Undo move.
            if has_occ {
                b.undo_move_with_occ(&undo, Some(backend.occ()));
            } else {
                b.undo_move_no_features(&undo);
            }
        }

        if has_occ {
            core::mem::swap(&mut b.feat_id_abs, scratch_feat);
            scratch_feat.clear();
        }

        (
            ExpandPlan {
                terminal_value: None,
                children,
            },
            used.min(budget_left),
        )
    }

    fn plan_solve_or_expand<B: AlphaBetaBackend>(
        backend: &mut B,
        cfg: DerivativeConfig,
        leaf: &TreeNode,
        eval_goal: Score,
        alpha: Score,
        beta: Score,
        budget_left: u64,
        scratch_feat: &mut Vec<u16>,
    ) -> UpdatePlan {
        // Proof budget.
        let mut proof_budget = max(cfg.solve_min_nodes, budget_left.min(cfg.solve_max_nodes));
        proof_budget = min(proof_budget, budget_left);

        // Window intersect with leaf strong bounds.
        let alpha2 = max(alpha, leaf.lower);
        let beta2 = min(beta, leaf.upper);
        let (alpha2, beta2) = if alpha2 < beta2 {
            (alpha2, beta2)
        } else {
            (leaf.lower, leaf.upper)
        };

        let mut b = leaf.to_board();
        let out = backend.exact_search_with_limits(
            &mut b,
            alpha2,
            beta2,
            SearchLimits {
                max_nodes: Some(proof_budget),
            },
        );

        let used = out.nodes.min(budget_left);
        if out.aborted {
            let budget_after = budget_left.saturating_sub(used);
            let (plan, u2) = plan_expand(backend, cfg, leaf, eval_goal, budget_after, scratch_feat);
            return UpdatePlan::Expand {
                plan,
                used: used.saturating_add(u2).min(budget_left),
            };
        }

        UpdatePlan::Solve {
            out,
            alpha2,
            beta2,
            used,
        }
    }

    fn commit_expand(
        arena: &mut TreeNodeSupplier,
        cfg: DerivativeConfig,
        node_id: NodeId,
        plan: ExpandPlan,
        weak_lower: Score,
        weak_upper: Score,
    ) -> Result<(), ArenaError> {
        // Only expand if still a leaf.
        if !arena.node(node_id).is_leaf {
            return Ok(());
        }

        // Terminal?
        if let Some(v) = plan.terminal_value {
            let n = arena.node_mut(node_id);
            n.is_terminal = true;
            n.is_leaf = true;
            n.lower = v;
            n.upper = v;
            n.leaf_eval = v;
            n.est = v;
            return Ok(());
        }

        // Mark internal.
        {
            let n = arena.node_mut(node_id);
            n.is_leaf = false;
            n.child_head = 0;
            n.child_count = 0;
        }

        for c in plan.children {
            let (child_id, is_new) = arena.get_or_create(c.hash, c.depth, c.bits, c.side, c.empty_count)?;
            arena.add_child(node_id, c.mv, child_id)?;
            arena.add_father(child_id, node_id)?;

            if is_new {
                let child = arena.node_mut(child_id);
                child.lower = score_min();
                child.upper = score_max();
                child.leaf_eval = c.quick_eval;
                child.est = c.quick_eval.clamp(child.lower, child.upper);
                child.eval_depth = 0;
                child.has_eval = true;

                // Initialize weak window around the quick eval, clamped into global.
                let (wlo, whi) = weak_for_depth_global(weak_lower, weak_upper, child.depth);
                let radius = max(cfg.weak_min_half_width, (child.upper - child.lower).abs() / 8);
                let local_lo = c.quick_eval.saturating_sub(radius);
                let local_hi = c.quick_eval.saturating_add(radius);

                let mut ww_lo = max(child.lower, max(wlo, local_lo));
                let mut ww_hi = min(child.upper, min(whi, local_hi));
                if ww_lo >= ww_hi {
                    ww_lo = child.lower;
                    ww_hi = child.upper;
                }
                child.weak_lower = ww_lo;
                child.weak_upper = ww_hi;

                // If we have a non-aborted seed, overwrite quick eval.
                if let Some((s, d)) = c.seed {
                    child.leaf_eval = s;
                    child.est = s.clamp(child.lower, child.upper);
                    child.eval_depth = d;
                }
            }
        }

        arena.update_node_from_children(node_id);
        Ok(())
    }

    fn commit_solve(arena: &mut TreeNodeSupplier, leaf_id: NodeId, out: SearchOutcome, alpha2: Score, beta2: Score) {
        let n = arena.node_mut(leaf_id);
        if out.score <= alpha2 {
            n.upper = min(n.upper, out.score);
        } else if out.score >= beta2 {
            n.lower = max(n.lower, out.score);
        } else {
            n.lower = max(n.lower, out.score);
            n.upper = min(n.upper, out.score);
        }

        n.leaf_eval = out.score;
        n.est = out.score.clamp(n.lower, n.upper);
        if n.lower >= n.upper {
            n.lower = out.score;
            n.upper = out.score;
        }
        n.n_updates = n.n_updates.saturating_add(1);
    }

    fn worker_loop<B: AlphaBetaBackend>(shared: Arc<SharedState>, mut backend: B) {
        let mut scratch_feat: Vec<u16> = Vec::with_capacity(N_PATTERN_FEATURES);

        loop {
            if shared.should_stop_fast() {
                break;
            }

            // Quick root-solved check (requires lock).
            {
                let arena = shared.arena.lock().unwrap();
                if arena.node(shared.root).is_solved() {
                    break;
                }
            }

            // Maintain global weak window.
            shared.maybe_update_global_weak_bounds();

            // Pick + lock a leaf.
            let sel = match shared.try_select_leaf() {
                Some(s) => s,
                None => continue,
            };

            // Take one iteration slot.
            let it = shared.iterations.fetch_add(1, Ordering::Relaxed) + 1;
            if it > shared.cfg.max_iterations {
                let mut arena = shared.arena.lock().unwrap();
                finalize_leaf_lock(&mut arena, &sel);
                break;
            }

            let budget_left = shared.budget_left();
            if budget_left == 0 {
                let mut arena = shared.arena.lock().unwrap();
                finalize_leaf_lock(&mut arena, &sel);
                break;
            }

            // Plan the update outside the arena lock.
            let plan = if to_be_solved_snapshot(shared.cfg, &sel.leaf_snapshot) {
                plan_solve_or_expand(
                    &mut backend,
                    shared.cfg,
                    &sel.leaf_snapshot,
                    sel.eval_goal,
                    sel.alpha,
                    sel.beta,
                    budget_left,
                    &mut scratch_feat,
                )
            } else {
                let (plan, used) = plan_expand(
                    &mut backend,
                    shared.cfg,
                    &sel.leaf_snapshot,
                    sel.eval_goal,
                    budget_left,
                    &mut scratch_feat,
                );
                UpdatePlan::Expand { plan, used }
            };

            let weak_lower = shared.weak_lower.load(Ordering::Relaxed);
            let weak_upper = shared.weak_upper.load(Ordering::Relaxed);

            // Commit under lock.
            let mut step_used: u64 = 0;
            {
                let mut arena = shared.arena.lock().unwrap();
                match plan {
                    UpdatePlan::Solve { out, alpha2, beta2, used } => {
                        commit_solve(&mut arena, sel.leaf, out, alpha2, beta2);
                        step_used = used;
                    }
                    UpdatePlan::Expand { plan, used } => {
                        if let Err(ArenaError::Full) = commit_expand(&mut arena, shared.cfg, sel.leaf, plan, weak_lower, weak_upper) {
                            shared.arena_full.store(true, Ordering::Relaxed);
                            step_used = 0;
                        } else {
                            step_used = used;
                        }
                    }
                }

                if step_used > 0 {
                    arena.propagate_descendants(sel.leaf, step_used);
                    arena.update_fathers_from_child(sel.leaf);
                }

                finalize_leaf_lock(&mut arena, &sel);
            }

            if step_used > 0 {
                shared.nodes_used.fetch_add(step_used, Ordering::Relaxed);
            }

            if shared.should_stop_fast() {
                break;
            }
        }
    }

    pub(super) fn evaluate_derivative_mt<B, F>(
        ev: &mut DerivativeEvaluator,
        make_backend: F,
        root_board: &Board,
        mut lower: Score,
        mut upper: Score,
        limits: SearchLimits,
    ) -> DerivativeResult
    where
        B: AlphaBetaBackend + Send + 'static,
        F: Fn(usize, usize) -> B + Send + Sync,
    {
        let avail = rayon::current_num_threads().max(1);
        let wanted = if ev.cfg.num_threads == 0 { avail } else { ev.cfg.num_threads.max(1).min(avail) };
        let n_threads = wanted.max(1);

        // Fallback to single-threaded if requested.
        if n_threads <= 1 {
            let mut backend = make_backend(0, 1);
            return ev.evaluate_with_bounds_backend(&mut backend, root_board, lower, upper, limits);
        }

        // Normalize bounds (same as single-thread).
        if lower > upper {
            core::mem::swap(&mut lower, &mut upper);
        }
        lower = lower.clamp(score_min(), score_max());
        upper = upper.clamp(score_min(), score_max());
        if lower > upper {
            lower = score_min();
            upper = score_max();
        }

        ev.arena.reset();
        ev.total_nodes_used = 0;
        ev.iterations = 0;

        ev.root_lower = lower;
        ev.root_upper = upper;
        ev.weak_lower = lower;
        ev.weak_upper = upper;

        if ev.root_lower > ev.root_upper {
            core::mem::swap(&mut ev.root_lower, &mut ev.root_upper);
            core::mem::swap(&mut ev.weak_lower, &mut ev.weak_upper);
        }

        // Root node init.
        let mut backend_init = make_backend(0, n_threads);

        let root_hash = root_board.hash;
        let root_bits = [root_board.player, root_board.opponent];
        let root_side = root_board.side;
        let root_empty = root_board.empty_count;

        let (root_id, _is_new) = match ev.arena.get_or_create(root_hash, 0, root_bits, root_side, root_empty) {
            Ok(v) => v,
            Err(_) => {
                return DerivativeResult {
                    best_move: PASS,
                    estimate: 0,
                    lower,
                    upper,
                    weak_lower: lower,
                    weak_upper: upper,
                    nodes_used: 0,
                    iterations: 0,
                    tree_nodes: 0,
                    status: DerivativeStatus::ArenaFull,
                };
            }
        };
        ev.root = root_id;

        {
            let n = ev.arena.node_mut(root_id);
            n.lower = lower;
            n.upper = upper;
            n.weak_lower = lower;
            n.weak_upper = upper;
            n.depth = 0;

            let b = root_board.clone();
            n.leaf_eval = backend_init.quick_eval(&b);
            n.est = n.leaf_eval.clamp(n.lower, n.upper);
            n.eval_depth = 0;
            n.has_eval = true;
        }

        let budget_total = limits.max_nodes.unwrap_or(u64::MAX);
        let mut status = DerivativeStatus::Budget;

        // Expand root once to build frontier.
        let used0 = match ev.expand_leaf(&mut backend_init, root_id, ev.root_eval_goal(), budget_total) {
            Ok(u) => u,
            Err(ArenaError::Full) => {
                status = DerivativeStatus::ArenaFull;
                0
            }
        };
        ev.total_nodes_used = ev.total_nodes_used.saturating_add(used0);
        ev.arena.propagate_descendants(root_id, used0);

        if ev.arena.node(root_id).is_terminal && ev.arena.node(root_id).is_solved() {
            status = DerivativeStatus::NoMoves;
        }

        // If we already stopped on root expansion, return without spawning.
        if status != DerivativeStatus::Budget || ev.total_nodes_used >= budget_total || ev.arena.node(root_id).is_solved() {
            if ev.arena.node(root_id).is_solved() {
                status = DerivativeStatus::Solved;
            } else if ev.total_nodes_used >= budget_total {
                status = DerivativeStatus::Budget;
            }

            let (best_move, estimate) = ev.best_move_and_estimate();
            let root = ev.arena.node(root_id);
            return DerivativeResult {
                best_move,
                estimate,
                lower: root.lower,
                upper: root.upper,
                weak_lower: ev.weak_lower,
                weak_upper: ev.weak_upper,
                nodes_used: ev.total_nodes_used,
                iterations: ev.iterations,
                tree_nodes: ev.arena.num_nodes(),
                status,
            };
        }

        // Move the arena into shared state without reallocating.
        let arena = core::mem::replace(&mut ev.arena, TreeNodeSupplier::placeholder(ev.cfg));

        let shared = Arc::new(SharedState {
            cfg: ev.cfg,
            root: root_id,
            root_lower: ev.root_lower,
            root_upper: ev.root_upper,
            budget_total,
            arena: Mutex::new(arena),
            weak_lower: AtomicI32::new(ev.weak_lower),
            weak_upper: AtomicI32::new(ev.weak_upper),
            is_updating_weak: AtomicBool::new(false),
            nodes_used: AtomicU64::new(ev.total_nodes_used),
            iterations: AtomicU32::new(0),
            arena_full: AtomicBool::new(false),
        });

        // Spawn workers.
        (0..n_threads)
            .into_par_iter()
            .for_each(|tid| {
                let backend = make_backend(tid, n_threads);
                worker_loop(shared.clone(), backend);
            });

        // Reclaim shared state.
        let shared = Arc::try_unwrap(shared).expect("derivative_mt: shared Arc still in use");
        let arena = shared.arena.into_inner().unwrap();

        // Restore evaluator state.
        ev.arena = arena;
        ev.total_nodes_used = shared.nodes_used.load(Ordering::Relaxed);
        ev.iterations = shared.iterations.load(Ordering::Relaxed);
        ev.weak_lower = shared.weak_lower.load(Ordering::Relaxed);
        ev.weak_upper = shared.weak_upper.load(Ordering::Relaxed);

        // Final status.
        let mut final_status = DerivativeStatus::Budget;
        if shared.arena_full.load(Ordering::Relaxed) {
            final_status = DerivativeStatus::ArenaFull;
        } else if ev.arena.node(root_id).is_terminal && ev.arena.node(root_id).is_solved() {
            final_status = DerivativeStatus::NoMoves;
        } else if ev.arena.node(root_id).is_solved() {
            final_status = DerivativeStatus::Solved;
        } else if ev.iterations >= ev.cfg.max_iterations {
            final_status = DerivativeStatus::MaxIterations;
        } else if ev.total_nodes_used >= budget_total {
            final_status = DerivativeStatus::Budget;
        }

        let (best_move, estimate) = ev.best_move_and_estimate();
        let root = ev.arena.node(root_id);

        DerivativeResult {
            best_move,
            estimate,
            lower: root.lower,
            upper: root.upper,
            weak_lower: ev.weak_lower,
            weak_upper: ev.weak_upper,
            nodes_used: ev.total_nodes_used,
            iterations: ev.iterations,
            tree_nodes: ev.arena.num_nodes(),
            status: final_status,
        }
    }
}


