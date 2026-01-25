//! Fixed-depth alpha-beta search (Sensei-style), using Sonetto's EGEV2 eval.
//!
//! ## Performance contract
//! The hot path must **not** do `Board` apply/undo. Native Sensei runs its
//! alpha-beta on raw bitboards plus an incremental evaluator.
//!
//! This backend mirrors that structure:
//! - the recursive search operates on `(player_bits, opponent_bits, side, empties, disc_counts)`
//! - EGEV2 evaluation at leaves uses the already-maintained **absolute ternary pattern IDs**
//!   (`feat_id_abs`) via `score_disc_from_abs_ids`
//! - feature IDs are updated/rolled back per move via `apply_move_feat_ids` / `rollback_move_feat_ids`
//! - root entry points still accept a `Board` (for compatibility), but only as an immutable
//!   *source* of the initial bitboards / cached features.

use crate::board::{Board, Color};
use crate::coord::{Move, PASS};
use crate::derivative::AlphaBetaBackend;
use crate::eval::{score_disc, FeatureDefs, Weights, N_PATTERN_FEATURES};
use crate::features::occ::OccMap;
use crate::features::swap::SwapTables;
use crate::features::update::recompute_features_in_place;
use crate::movegen::legal_moves;
use crate::score::{game_over_scaled_from_bits, Score, SCALE};
use crate::stability::stable_disks_definitely;
use crate::search::{
    AnalyzeMode, AnalyzeTopNRequest, AnalyzeTopNResult, AnalyzeTopNStats, SearchLimits, SearchOutcome,
};

// Optional parallel root-split (Rayon).
#[cfg(feature = "parallel_rayon")]
use rayon::prelude::*;

use super::depth_one::{apply_move_feat_ids, rollback_move_feat_ids, score_disc_from_abs_ids};
use super::hash_map::{SenseiHashMap, MIN_SCORE};
use super::last_moves;
use super::move_iter::{gen_ordered_moves, DisproofCtx, MoveIteratorKind};
use std::sync::Arc;

const INF: Score = 1_000_000;

#[inline(always)]
fn use_sensei_hash(depth: u8, empty_count: u8, solve: bool) -> bool {
    // Sensei `UseHashMap`:
    //   depth >= (solve ? 10 : 3)  &&  empty_count >= 10
    // (engine/utils/constants.h)
    let depth_thresh = if solve { 10 } else { 3 };
    depth >= depth_thresh && empty_count >= 10
}

/// Bump TT moves to the front of the ordered move list (Sensei: best + second-best).
#[inline(always)]
fn bump_tt_moves(moves: &mut [Move; 64], flips: &mut [u64; 64], n: usize, tt_best: Move, tt_second: Move) {
    #[inline(always)]
    fn bring_to_front(moves: &mut [Move; 64], flips: &mut [u64; 64], n: usize, target: Move, front: usize) {
        if target == PASS || front >= n {
            return;
        }
        for i in front..n {
            if moves[i] == target {
                moves.swap(front, i);
                flips.swap(front, i);
                return;
            }
        }
    }

    bring_to_front(moves, flips, n, tt_best, 0);
    if tt_second != tt_best {
        bring_to_front(moves, flips, n, tt_second, 1);
    }
}

/// Sensei-style fixed-depth alpha-beta backend.
#[derive(Clone)]
pub struct SenseiAlphaBeta {
    weights: Weights,
    feats: FeatureDefs,
    swap: SwapTables,
    occ: OccMap,

    // Sensei-style direct-mapped hash table (transposition cache).
    hash_map: Arc<SenseiHashMap>,

    // Per-search mutable state.
    nodes: u64,
    aborted: bool,
    limits: SearchLimits,
}

impl SenseiAlphaBeta {
    /// Create a new backend instance.
    ///
    /// `hash_size_mb` controls the size of the Sensei-style direct-mapped hash map.
    pub fn new_with_hash_size_mb(
        hash_size_mb: usize,
        weights: Weights,
        feats: FeatureDefs,
        swap: SwapTables,
        occ: OccMap,
    ) -> Self {
        Self {
            weights,
            feats,
            swap,
            occ,
            hash_map: Arc::new(SenseiHashMap::new_mb(hash_size_mb)),
            nodes: 0,
            aborted: false,
            limits: SearchLimits::default(),
        }
    }

    /// Backwards-compatible constructor (defaults to 16MB hash).
    pub fn new(weights: Weights, feats: FeatureDefs, swap: SwapTables, occ: OccMap) -> Self {
        Self::new_with_hash_size_mb(16, weights, feats, swap, occ)
    }

    /// Last search node count.
    pub fn last_nodes(&self) -> u64 {
        self.nodes
    }

    /// Whether the last search aborted due to `SearchLimits`.
    pub fn last_aborted(&self) -> bool {
        self.aborted
    }

    /// Fixed-depth best-move search.
    pub fn best_move(&mut self, board: &mut Board, depth: u8, limits: SearchLimits) -> (Move, Score) {
        self.nodes = 0;
        self.aborted = false;
        self.limits = limits;

        self.ensure_pattern_features(board);

        let side = board.side;
        let player = board.player;
        let opponent = board.opponent;
        let empty_count = board.empty_count;
        let player_discs = player.count_ones() as u8;
        let opponent_discs = opponent.count_ones() as u8;

        let mut feat: [u16; N_PATTERN_FEATURES] = [0u16; N_PATTERN_FEATURES];
        feat.copy_from_slice(&board.feat_id_abs[..N_PATTERN_FEATURES]);

        self.root_search_window_bits(
            player,
            opponent,
            side,
            empty_count,
            player_discs,
            opponent_discs,
            depth,
            -INF,
            INF,
            &mut feat,
        )
    }

    // ---------------------------------------------------------------------
    // Top-N analysis (sequential)
    // ---------------------------------------------------------------------

    pub fn analyze_top_n(&mut self, board: &mut Board, request: AnalyzeTopNRequest) -> AnalyzeTopNResult {
        self.nodes = 0;
        self.aborted = false;
        self.limits = SearchLimits::default();

        self.ensure_pattern_features(board);

        let depth = match request.mode {
            AnalyzeMode::Midgame { depth } => depth,
            AnalyzeMode::Exact => {
                // This backend supports solve-to-end separately. Keep legacy behavior here:
                // fall back to a moderate fixed depth.
                8
            }
        };

        let top_n = request.top_n as usize;

        let side = board.side;
        let player = board.player;
        let opponent = board.opponent;
        let empty_count = board.empty_count;
        let player_discs = player.count_ones() as u8;
        let opponent_discs = opponent.count_ones() as u8;

        let mut feat: [u16; N_PATTERN_FEATURES] = [0u16; N_PATTERN_FEATURES];
        feat.copy_from_slice(&board.feat_id_abs[..N_PATTERN_FEATURES]);

        // Root move ordering.
        let mut moves = [PASS; 64];
        let mut flips = [0u64; 64];
        let kind = pick_iterator_kind(depth, false, false);
        let ctx = Some(self.make_disproof_ctx(side, empty_count, player_discs, opponent_discs, feat.as_mut_slice()));
        let n = gen_ordered_moves(kind, player, opponent, 0, INF, ctx, &mut moves, &mut flips);

        if n == 0 {
            let stats = AnalyzeTopNStats::Fixed {
                nodes_used: self.nodes,
                aborted: self.aborted,
            };
            return AnalyzeTopNResult { pairs: Vec::new(), stats };
        }

        let child_depth = depth.saturating_sub(1);
        let mut pairs: Vec<(Move, Score)> = Vec::with_capacity(n);

        for i in 0..n {
            if self.aborted {
                break;
            }
            let mv = moves[i];
            let fl = flips[i];
            let mv_bit = 1u64 << mv;
            let fcnt = fl.count_ones() as u8;

            // Apply move to feature IDs.
            apply_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            // Child state (swap roles).
            let child_player = opponent & !fl;
            let child_opponent = player | mv_bit | fl;
            let child_side = side.other();
            let child_empty = empty_count - 1;
            let child_player_discs = opponent_discs - fcnt;
            let child_opponent_discs = player_discs + 1 + fcnt;

            let score = -self.negamax_bits(
                child_player,
                child_opponent,
                child_side,
                child_empty,
                child_player_discs,
                child_opponent_discs,
                child_depth,
                -INF,
                INF,
                mv_bit | fl,
                &mut feat,
            );

            // Roll back feature IDs.
            rollback_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            pairs.push((mv, score));
        }

        // Sort descending by score.
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        if pairs.len() > top_n {
            pairs.truncate(top_n);
        }

        let stats = AnalyzeTopNStats::Fixed {
            nodes_used: self.nodes,
            aborted: self.aborted,
        };

        AnalyzeTopNResult { pairs, stats }
    }

    // ---------------------------------------------------------------------
    // Parallel root-split Top-N analysis (Rayon)
    // ---------------------------------------------------------------------

    #[cfg(feature = "parallel_rayon")]
    pub fn analyze_top_n_parallel(&mut self, board: &mut Board, request: AnalyzeTopNRequest) -> AnalyzeTopNResult {
        let avail_threads = rayon::current_num_threads();
        if avail_threads <= 1 {
            return self.analyze_top_n(board, request);
        }

        self.nodes = 0;
        self.aborted = false;
        self.limits = SearchLimits::default();

        self.ensure_pattern_features(board);

        let depth = match request.mode {
            AnalyzeMode::Midgame { depth } => depth,
            AnalyzeMode::Exact => 8,
        };

        let top_n = request.top_n as usize;

        let side = board.side;
        let player = board.player;
        let opponent = board.opponent;
        let empty_count = board.empty_count;
        let player_discs = player.count_ones() as u8;
        let opponent_discs = opponent.count_ones() as u8;

        let mut feat_root: [u16; N_PATTERN_FEATURES] = [0u16; N_PATTERN_FEATURES];
        feat_root.copy_from_slice(&board.feat_id_abs[..N_PATTERN_FEATURES]);

        // Root move ordering.
        let mut moves = [PASS; 64];
        let mut flips = [0u64; 64];
        let kind = pick_iterator_kind(depth, false, false);
        let ctx = Some(self.make_disproof_ctx(side, empty_count, player_discs, opponent_discs, feat_root.as_mut_slice()));
        let n = gen_ordered_moves(kind, player, opponent, 0, INF, ctx, &mut moves, &mut flips);

        if n == 0 {
            let stats = AnalyzeTopNStats::Fixed {
                nodes_used: self.nodes,
                aborted: self.aborted,
            };
            return AnalyzeTopNResult { pairs: Vec::new(), stats };
        }

        let t = avail_threads.max(1).min(n);
        if t <= 1 || n <= 1 {
            return self.analyze_top_n(board, request);
        }

        let weights = self.weights.clone();
        let feats = self.feats.clone();
        let swap = self.swap.clone();
        let occ = self.occ.clone();

        let child_depth = depth.saturating_sub(1);

        let parts: Vec<(Vec<(Move, Score)>, u64)> = (0..t)
            .into_par_iter()
            .map(|tid| {
                let mut worker = SenseiAlphaBeta::new(weights.clone(), feats.clone(), swap.clone(), occ.clone());
                worker.nodes = 0;
                worker.aborted = false;
                worker.limits = SearchLimits::default();

                let mut feat: [u16; N_PATTERN_FEATURES] = feat_root;
                let mut out: Vec<(Move, Score)> = Vec::with_capacity((n + t - 1) / t);

                for idx in (tid..n).step_by(t) {
                    let mv = moves[idx];
                    let fl = flips[idx];
                    let mv_bit = 1u64 << mv;
                    let fcnt = fl.count_ones() as u8;

                    apply_move_feat_ids(feat.as_mut_slice(), &worker.occ, mv, fl, side);

                    let child_player = opponent & !fl;
                    let child_opponent = player | mv_bit | fl;
                    let child_side = side.other();
                    let child_empty = empty_count - 1;
                    let child_player_discs = opponent_discs - fcnt;
                    let child_opponent_discs = player_discs + 1 + fcnt;

                    let score = -worker.negamax_bits(
                        child_player,
                        child_opponent,
                        child_side,
                        child_empty,
                        child_player_discs,
                        child_opponent_discs,
                        child_depth,
                        -INF,
                        INF,
                        mv_bit | fl,
                        &mut feat,
                    );

                    rollback_move_feat_ids(feat.as_mut_slice(), &worker.occ, mv, fl, side);

                    out.push((mv, score));
                }

                (out, worker.nodes)
            })
            .collect();

        let mut pairs: Vec<(Move, Score)> = Vec::with_capacity(n);
        let mut total_nodes: u64 = 0;
        for (mut v, nn) in parts {
            total_nodes = total_nodes.wrapping_add(nn);
            pairs.append(&mut v);
        }
        self.nodes = total_nodes;

        // Deterministic ordering: score desc, then move asc.
        pairs.sort_by(|(m1, s1), (m2, s2)| s2.cmp(s1).then_with(|| m1.cmp(m2)));
        if pairs.len() > top_n {
            pairs.truncate(top_n);
        }

        let stats = AnalyzeTopNStats::Fixed {
            nodes_used: self.nodes,
            aborted: self.aborted,
        };

        AnalyzeTopNResult { pairs, stats }
    }

    // ---------------------------------------------------------------------
    // Core eval / ordering helpers
    // ---------------------------------------------------------------------

    #[inline(always)]
    fn eval_from_abs(&self, side: Color, empty_count: u8, player_discs: u8, feat: &[u16; N_PATTERN_FEATURES]) -> Score {
        (score_disc_from_abs_ids(side, empty_count, player_discs, feat.as_slice(), &self.weights) as Score) * SCALE
    }

    /// Build the optional disproof-number ordering context.
    #[inline(always)]
    fn make_disproof_ctx<'a>(
        &'a self,
        side: Color,
        empty_count: u8,
        player_discs: u8,
        opponent_discs: u8,
        feat_id_abs: &'a mut [u16],
    ) -> DisproofCtx<'a> {
        DisproofCtx {
            side,
            empty_count,
            player_discs,
            opponent_discs,
            feat_id_abs,
            occ: &self.occ,
            weights: &self.weights,
        }
    }

    fn ensure_pattern_features(&self, board: &mut Board) {
        if board.feat_id_abs.len() != N_PATTERN_FEATURES {
            board.feat_id_abs.resize(N_PATTERN_FEATURES, 0);
            board.feat_is_pattern_ids = false;
        }
        recompute_features_in_place(board, &self.occ);
    }

    #[inline(always)]
    fn maybe_abort(&mut self) -> bool {
        if let Some(max_nodes) = self.limits.max_nodes {
            if self.nodes >= max_nodes {
                self.aborted = true;
                return true;
            }
        }
        false
    }

    // ---------------------------------------------------------------------
    // Root search entry points (bitboard core)
    // ---------------------------------------------------------------------

    fn root_search_window_bits(
        &mut self,
        player: u64,
        opponent: u64,
        side: Color,
        empty_count: u8,
        player_discs: u8,
        opponent_discs: u8,
        depth: u8,
        mut alpha: Score,
        beta: Score,
        feat: &mut [u16; N_PATTERN_FEATURES],
    ) -> (Move, Score) {
        let mut moves = [PASS; 64];
        let mut flips = [0u64; 64];

        let kind = pick_iterator_kind(depth, false, false);
        let ctx = Some(self.make_disproof_ctx(side, empty_count, player_discs, opponent_discs, feat.as_mut_slice()));
        let n = gen_ordered_moves(kind, player, opponent, 0, beta, ctx, &mut moves, &mut flips);

        if n == 0 {
            // No legal moves => pass or terminal.
            let opp_moves = legal_moves(opponent, player);
            if opp_moves == 0 {
                return (PASS, game_over_scaled_from_bits(player, opponent));
            }

            // Pass: swap roles. Pass does NOT consume depth.
            let score = -self.negamax_bits(
                opponent,
                player,
                side.other(),
                empty_count,
                opponent_discs,
                player_discs,
                depth,
                -beta,
                -alpha,
                0,
                feat,
            );
            return (PASS, score);
        }

        let mut best_move = moves[0];
        let mut best_score = -INF;

        for i in 0..n {
            if self.aborted {
                break;
            }

            let mv = moves[i];
            let mv_bit = 1u64 << mv;
            let fl = flips[i];
            let fcnt = fl.count_ones() as u8;

            // Update features.
            apply_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            // Child state.
            let child_player = opponent & !fl;
            let child_opponent = player | mv_bit | fl;
            let child_side = side.other();
            let child_empty = empty_count - 1;
            let child_player_discs = opponent_discs - fcnt;
            let child_opponent_discs = player_discs + 1 + fcnt;

            let score = -self.negamax_bits(
                child_player,
                child_opponent,
                child_side,
                child_empty,
                child_player_discs,
                child_opponent_discs,
                depth.saturating_sub(1),
                -beta,
                -alpha,
                mv_bit | fl,
                feat,
            );

            // Roll back features.
            rollback_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            if self.aborted {
                break;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
            }

            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }

        (best_move, best_score)
    }

    fn solve_root_window_bits(
        &mut self,
        player: u64,
        opponent: u64,
        side: Color,
        empty_count: u8,
        player_discs: u8,
        opponent_discs: u8,
        mut alpha: Score,
        beta: Score,
        feat: &mut [u16; N_PATTERN_FEATURES],
    ) -> (Move, Score) {
        let mut moves = [PASS; 64];
        let mut flips = [0u64; 64];

        let depth_hint = empty_count;
        let kind = pick_iterator_kind(depth_hint, true, false);
        let ctx = Some(self.make_disproof_ctx(side, empty_count, player_discs, opponent_discs, feat.as_mut_slice()));
        let n = gen_ordered_moves(kind, player, opponent, 0, beta, ctx, &mut moves, &mut flips);

        if n == 0 {
            let opp_moves = legal_moves(opponent, player);
            if opp_moves == 0 {
                return (PASS, game_over_scaled_from_bits(player, opponent));
            }

            let score = -self.negamax_solve_bits(
                opponent,
                player,
                side.other(),
                empty_count,
                opponent_discs,
                player_discs,
                -beta,
                -alpha,
                0,
                feat,
            );
            return (PASS, score);
        }

        let mut best_move = moves[0];
        let mut best_score = -INF;

        for i in 0..n {
            if self.aborted {
                break;
            }

            let mv = moves[i];
            let mv_bit = 1u64 << mv;
            let fl = flips[i];
            let fcnt = fl.count_ones() as u8;

            apply_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            let child_player = opponent & !fl;
            let child_opponent = player | mv_bit | fl;
            let child_side = side.other();
            let child_empty = empty_count - 1;
            let child_player_discs = opponent_discs - fcnt;
            let child_opponent_discs = player_discs + 1 + fcnt;

            let score = -self.negamax_solve_bits(
                child_player,
                child_opponent,
                child_side,
                child_empty,
                child_player_discs,
                child_opponent_discs,
                -beta,
                -alpha,
                mv_bit | fl,
                feat,
            );

            rollback_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            if self.aborted {
                break;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
            }

            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }

        (best_move, best_score)
    }

    // ---------------------------------------------------------------------
    // Midgame negamax alpha-beta (pure bitboard recursion)
    // ---------------------------------------------------------------------

    fn negamax_bits(
        &mut self,
        player: u64,
        opponent: u64,
        side: Color,
        empty_count: u8,
        player_discs: u8,
        opponent_discs: u8,
        depth: u8,
        mut alpha: Score,
        beta: Score,
        last_flip_incl_move: u64,
        feat: &mut [u16; N_PATTERN_FEATURES],
    ) -> Score {
        self.nodes = self.nodes.wrapping_add(1);
        if self.maybe_abort() {
            return 0;
        }

        if depth == 0 {
            return self.eval_from_abs(side, empty_count, player_discs, feat);
        }

        let lower0 = alpha;
        let mut beta = beta;

        // Compute depth-zero eval only when we actually need it.
        // - always needed for depth==1 (Sensei depth-one blend)
        // - needed for "unlikely" detection when depth<=13 and stability is not enough to decide
        let mut depth_zero_eval: Score = 0;
        let mut have_depth_zero_eval = false;

        // --- Sensei-style hash probe (direct mapped) ---
        let mut tt_best: Move = PASS;
        let mut tt_second: Move = PASS;
        if use_sensei_hash(depth, empty_count, false) {
            if let Some(e) = self.hash_map.get(player, opponent) {
                tt_best = e.best_move;
                tt_second = e.second_best_move;

                if e.depth >= depth {
                    if e.lower == e.upper {
                        return e.lower;
                    }
                    if e.lower >= beta {
                        return e.lower;
                    }
                    if e.upper <= alpha {
                        return e.upper;
                    }

                    alpha = alpha.max(e.lower);
                    beta = beta.min(e.upper);
                }
            }
        }

        // --- Stability cutoff (Sensei `UseStabilityCutoff(depth > 3)`) ---
        let mut stability_cutoff_upper = beta;
        if depth > 3 {
            let stable_opp = stable_disks_definitely(opponent, player);
            let stable_opp_cnt = stable_opp.count_ones() as i32;
            stability_cutoff_upper = (64 - 2 * stable_opp_cnt) * SCALE;
            if stability_cutoff_upper <= alpha {
                return stability_cutoff_upper;
            }
        }

        // --- "Unlikely" move ordering hint (Sensei) ---
        let unlikely = if depth <= 13 {
            if stability_cutoff_upper < alpha + 15 * SCALE {
                true
            } else {
                depth_zero_eval = self.eval_from_abs(side, empty_count, player_discs, feat);
                have_depth_zero_eval = true;
                depth_zero_eval < alpha - 5 * SCALE
            }
        } else {
            false
        };

        // Generate ordered moves.
        let mut moves = [PASS; 64];
        let mut flips = [0u64; 64];

        let kind = pick_iterator_kind(depth, false, unlikely);
        let ctx = Some(self.make_disproof_ctx(side, empty_count, player_discs, opponent_discs, feat.as_mut_slice()));
        let n = gen_ordered_moves(kind, player, opponent, last_flip_incl_move, beta, ctx, &mut moves, &mut flips);

        if n == 0 {
            let opp_moves = legal_moves(opponent, player);
            if opp_moves == 0 {
                return game_over_scaled_from_bits(player, opponent);
            }
            // Pass does NOT consume depth.
            return -self.negamax_bits(
                opponent,
                player,
                side.other(),
                empty_count,
                opponent_discs,
                player_discs,
                depth,
                -beta,
                -alpha,
                0,
                feat,
            );
        }

        // Prefer TT best/second-best moves.
        bump_tt_moves(&mut moves, &mut flips, n, tt_best, tt_second);

        let mut best_score: Score = MIN_SCORE;
        let mut best_move: Move = moves[0];
        let mut second_best_score: Score = MIN_SCORE;
        let mut second_best_move: Move = PASS;

        // If depth==1, we need depth_zero_eval.
        if depth == 1 && !have_depth_zero_eval {
            depth_zero_eval = self.eval_from_abs(side, empty_count, player_discs, feat);
            have_depth_zero_eval = true;
        }

        for i in 0..n {
            if self.aborted {
                break;
            }

            let mv = moves[i];
            let mv_bit = 1u64 << mv;
            let fl = flips[i];
            let fcnt = fl.count_ones() as u8;

            apply_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            let child_player = opponent & !fl;
            let child_opponent = player | mv_bit | fl;
            let child_side = side.other();
            let child_empty = empty_count - 1;
            let child_player_discs = opponent_discs - fcnt;
            let child_opponent_discs = player_discs + 1 + fcnt;

            let score = if depth == 1 {
                // Native Sensei depth=1 shortcut:
                // current_eval = (E0*kW0 - E1*kW1) / (kW0+kW1), with kW0=1,kW1=2.
                let child_eval = self.eval_from_abs(child_side, child_empty, child_player_discs, feat);
                (depth_zero_eval - child_eval * 2) / 3
            } else {
                -self.negamax_bits(
                    child_player,
                    child_opponent,
                    child_side,
                    child_empty,
                    child_player_discs,
                    child_opponent_discs,
                    depth - 1,
                    -beta,
                    -alpha,
                    mv_bit | fl,
                    feat,
                )
            };

            rollback_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            if self.aborted {
                break;
            }

            if score > best_score {
                second_best_score = best_score;
                second_best_move = best_move;

                best_score = score;
                best_move = mv;
            } else if score > second_best_score {
                second_best_score = score;
                second_best_move = mv;
            }

            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }

        if use_sensei_hash(depth, empty_count, false) {
            self.hash_map.update(player, opponent, depth, best_score, lower0, beta, best_move, second_best_move);
        }

        best_score
    }

    // ---------------------------------------------------------------------
    // Exact solve-to-end negamax alpha-beta (pure bitboard recursion)
    // ---------------------------------------------------------------------

    fn negamax_solve_bits(
        &mut self,
        player: u64,
        opponent: u64,
        side: Color,
        empty_count: u8,
        player_discs: u8,
        opponent_discs: u8,
        mut alpha: Score,
        beta: Score,
        last_flip_incl_move: u64,
        feat: &mut [u16; N_PATTERN_FEATURES],
    ) -> Score {
        self.nodes = self.nodes.wrapping_add(1);
        if self.maybe_abort() {
            return 0;
        }

        let depth_hint = empty_count;

        // -----------------------------------------------------------------
        // Native Sensei: "last moves" fast solver (<= 5 empties).
        // -----------------------------------------------------------------
        if depth_hint <= 5 {
            // Bounds in disc-diff units ([-66,66] in Sensei last-moves code).
            let lower_disc = (alpha / SCALE).clamp(last_moves::LESS_THAN_MIN_EVAL, 66);
            let upper_disc = (beta / SCALE).clamp(-66, 66);

            let mut visited: u64 = 0;

            let eval_disc = match depth_hint {
                0 => {
                    // Game over with no empties.
                    return game_over_scaled_from_bits(player, opponent);
                }
                1 => {
                    // Enumerate the only empty.
                    let empties = !(player | opponent);
                    let x = empties.trailing_zeros() as u8;
                    last_moves::eval_one_empty(x, player, opponent)
                }
                2 => {
                    let empties = !(player | opponent);
                    let x1 = empties.trailing_zeros() as u8;
                    let x2 = (empties & (empties - 1)).trailing_zeros() as u8;
                    last_moves::eval_two_empties(x1, x2, player, opponent, lower_disc, upper_disc, &mut visited)
                }
                3 => {
                    let empties = !(player | opponent);
                    let mut m = empties;
                    let x1 = m.trailing_zeros() as u8;
                    m &= m - 1;
                    let x2 = m.trailing_zeros() as u8;
                    m &= m - 1;
                    let x3 = m.trailing_zeros() as u8;
                    last_moves::eval_three_empties(x1, x2, x3, player, opponent, lower_disc, upper_disc, &mut visited)
                }
                4 => {
                    let empties = !(player | opponent);
                    let mut m = empties;
                    let x1 = m.trailing_zeros() as u8;
                    m &= m - 1;
                    let x2 = m.trailing_zeros() as u8;
                    m &= m - 1;
                    let x3 = m.trailing_zeros() as u8;
                    m &= m - 1;
                    let x4 = m.trailing_zeros() as u8;

                    // For direct 4-empties entry, we keep a neutral `swap=false`.
                    last_moves::eval_four_empties(x1, x2, x3, x4, player, opponent, lower_disc, upper_disc, false, last_flip_incl_move, &mut visited)
                }
                5 => {
                    // EvalFiveEmpties handles its own empties re-ordering.
                    last_moves::eval_five_empties(player, opponent, lower_disc, upper_disc, last_flip_incl_move, &mut visited)
                }
                _ => unreachable!(),
            };

            // Keep node accounting roughly consistent with the recursive solver.
            self.nodes = self.nodes.wrapping_add(visited);

            return eval_disc * SCALE;
        }

        let lower0 = alpha;
        let mut beta = beta;

        // --- Sensei-style hash probe (direct mapped) ---
        let mut tt_best: Move = PASS;
        let mut tt_second: Move = PASS;
        if use_sensei_hash(depth_hint, empty_count, true) {
            if let Some(e) = self.hash_map.get(player, opponent) {
                tt_best = e.best_move;
                tt_second = e.second_best_move;

                if e.depth >= depth_hint {
                    if e.lower == e.upper {
                        return e.lower;
                    }
                    if e.lower >= beta {
                        return e.lower;
                    }
                    if e.upper <= alpha {
                        return e.upper;
                    }

                    alpha = alpha.max(e.lower);
                    beta = beta.min(e.upper);
                }
            }
        }

        // --- Stability cutoff (Sensei `UseStabilityCutoff(depth > 3)`) ---
        let mut stability_cutoff_upper = beta;
        if depth_hint > 3 {
            let stable_opp = stable_disks_definitely(opponent, player);
            let stable_opp_cnt = stable_opp.count_ones() as i32;
            stability_cutoff_upper = (64 - 2 * stable_opp_cnt) * SCALE;
            if stability_cutoff_upper <= alpha {
                return stability_cutoff_upper;
            }
        }

        // --- "Unlikely" hint only around 12-13 empties (Sensei) ---
        let unlikely = if depth_hint >= 12 && depth_hint <= 13 {
            if stability_cutoff_upper < alpha + 15 * SCALE {
                true
            } else {
                let depth_zero_eval = self.eval_from_abs(side, empty_count, player_discs, feat);
                depth_zero_eval < alpha - 5 * SCALE
            }
        } else {
            false
        };

        // Generate ordered moves.
        let mut moves = [PASS; 64];
        let mut flips = [0u64; 64];

        let kind = pick_iterator_kind(depth_hint, true, unlikely);
        let ctx = Some(self.make_disproof_ctx(side, empty_count, player_discs, opponent_discs, feat.as_mut_slice()));
        let n = gen_ordered_moves(kind, player, opponent, last_flip_incl_move, beta, ctx, &mut moves, &mut flips);

        if n == 0 {
            let opp_moves = legal_moves(opponent, player);
            if opp_moves == 0 {
                return game_over_scaled_from_bits(player, opponent);
            }

            // Pass does NOT consume depth in solve-to-end.
            return -self.negamax_solve_bits(
                opponent,
                player,
                side.other(),
                empty_count,
                opponent_discs,
                player_discs,
                -beta,
                -alpha,
                0,
                feat,
            );
        }

        bump_tt_moves(&mut moves, &mut flips, n, tt_best, tt_second);

        let mut best_score = MIN_SCORE;
        let mut best_move: Move = moves[0];
        let mut second_best_score = MIN_SCORE;
        let mut second_best_move: Move = PASS;

        for i in 0..n {
            if self.aborted {
                break;
            }

            let mv = moves[i];
            let mv_bit = 1u64 << mv;
            let fl = flips[i];
            let fcnt = fl.count_ones() as u8;

            apply_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            let child_player = opponent & !fl;
            let child_opponent = player | mv_bit | fl;
            let child_side = side.other();
            let child_empty = empty_count - 1;
            let child_player_discs = opponent_discs - fcnt;
            let child_opponent_discs = player_discs + 1 + fcnt;

            let score = -self.negamax_solve_bits(
                child_player,
                child_opponent,
                child_side,
                child_empty,
                child_player_discs,
                child_opponent_discs,
                -beta,
                -alpha,
                mv_bit | fl,
                feat,
            );

            rollback_move_feat_ids(feat.as_mut_slice(), &self.occ, mv, fl, side);

            if self.aborted {
                break;
            }

            if score > best_score {
                second_best_score = best_score;
                second_best_move = best_move;

                best_score = score;
                best_move = mv;
            } else if score > second_best_score {
                second_best_score = score;
                second_best_move = mv;
            }

            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }

        if use_sensei_hash(depth_hint, empty_count, true) {
            self.hash_map.update(player, opponent, depth_hint, best_score, lower0, beta, best_move, second_best_move);
        }

        best_score
    }
}

impl AlphaBetaBackend for SenseiAlphaBeta {
    #[inline(always)]
    fn occ(&self) -> &OccMap {
        &self.occ
    }

    #[inline(always)]
    fn quick_eval(&self, board: &Board) -> Score {
        // Align with Sonetto's derivative scheduler: use the disc-scale score.
        score_disc(board, &self.weights) * SCALE
    }

    #[inline]
    fn search_with_limits(
        &mut self,
        board: &mut Board,
        alpha: Score,
        beta: Score,
        depth: u8,
        limits: SearchLimits,
    ) -> SearchOutcome {
        self.nodes = 0;
        self.aborted = false;

        let old_limits = self.limits;
        self.limits = limits;

        self.ensure_pattern_features(board);

        let side = board.side;
        let player = board.player;
        let opponent = board.opponent;
        let empty_count = board.empty_count;
        let player_discs = player.count_ones() as u8;
        let opponent_discs = opponent.count_ones() as u8;

        let mut feat: [u16; N_PATTERN_FEATURES] = [0u16; N_PATTERN_FEATURES];
        feat.copy_from_slice(&board.feat_id_abs[..N_PATTERN_FEATURES]);

        let (best_move, score) = self.root_search_window_bits(
            player,
            opponent,
            side,
            empty_count,
            player_discs,
            opponent_discs,
            depth,
            alpha,
            beta,
            &mut feat,
        );

        let out = SearchOutcome {
            best_move,
            score,
            nodes: self.nodes,
            aborted: self.aborted,
        };

        self.limits = old_limits;
        out
    }

    #[inline]
    fn exact_search_with_limits(&mut self, board: &mut Board, alpha: Score, beta: Score, limits: SearchLimits) -> SearchOutcome {
        self.nodes = 0;
        self.aborted = false;

        let old_limits = self.limits;
        self.limits = limits;

        self.ensure_pattern_features(board);

        let side = board.side;
        let player = board.player;
        let opponent = board.opponent;
        let empty_count = board.empty_count;
        let player_discs = player.count_ones() as u8;
        let opponent_discs = opponent.count_ones() as u8;

        let mut feat: [u16; N_PATTERN_FEATURES] = [0u16; N_PATTERN_FEATURES];
        feat.copy_from_slice(&board.feat_id_abs[..N_PATTERN_FEATURES]);

        let (best_move, score) = self.solve_root_window_bits(
            player,
            opponent,
            side,
            empty_count,
            player_discs,
            opponent_discs,
            alpha,
            beta,
            &mut feat,
        );

        let out = SearchOutcome {
            best_move,
            score,
            nodes: self.nodes,
            aborted: self.aborted,
        };

        self.limits = old_limits;
        out
    }
}

#[inline(always)]
fn pick_iterator_kind(depth: u8, solve: bool, unlikely: bool) -> MoveIteratorKind {
    // Mirror native Sensei's `MoveIteratorOffset` schedule (engine/move_iter.cc).
    let unlikely = unlikely && depth <= 13;
    if unlikely {
        return if depth <= 9 {
            MoveIteratorKind::VeryQuick
        } else {
            MoveIteratorKind::Quick1
        };
    }

    if (solve && depth <= 8) || (!solve && depth <= 2) {
        return MoveIteratorKind::Quick1;
    }
    if (solve && depth <= 9) || (!solve && depth <= 4) {
        return MoveIteratorKind::Quick2;
    }

    if solve {
        if depth < 12 {
            MoveIteratorKind::MinimizeOpponentMoves
        } else {
            MoveIteratorKind::DisproofNumber
        }
    } else {
        MoveIteratorKind::MinimizeOpponentMoves
    }
}
