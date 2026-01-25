use sonetto_core::{
    board::Board,
    coord::PASS,
    derivative::{DerivativeConfig, DerivativeEvaluator},
    eval::Weights,
    eval_egev2::decode_egev2,
    features::{occ::OccMap, swap::SwapTables, update::push_moves_from_mask},
    movegen::legal_moves,
    search::{SearchLimits, Searcher},
    sensei_ab::SenseiAlphaBeta,
};

use std::time::{Duration, Instant};

const INF: i32 = 1_000_000;

fn main() {
    // Shared eval resources.
    let weights = try_load_weights();

    let feats = sonetto_core::features::FeatureDefs::load(
        "crates/sonetto_core/data/egbk3/features.json",
    )
    .expect("failed to load features");

    let occ = OccMap::new(&feats);
    let swap = SwapTables::new();

    // Engine instances.
    let mut sonetto = Searcher::new(16, weights.clone(), feats.clone(), swap.clone(), occ.clone());
    let mut sensei = SenseiAlphaBeta::new(weights.clone(), feats.clone(), swap.clone(), occ.clone());

    // Derivative scheduler (driving SenseiAB backend).
    let mut der_cfg = DerivativeConfig::default().with_tree_node_cap(120_000);
    // Fix seed depth to reduce variance.
    der_cfg.seed_depth_min = 4;
    der_cfg.seed_depth_max = 4;
    let mut deriv = DerivativeEvaluator::new(der_cfg);

    // Midgame test position.
    let base = random_midgame_board(12);
    println!("=== nps_compare ===");
    println!("empties: {}", base.empty_count);

    // Budgeted run (comparable across all three strategies).
    let depth: u8 = 10;
    let budget: u64 = 2_000_000;
    let limits = SearchLimits {
        max_nodes: Some(budget),
    };

    // Warm-up (JIT/ICache effects, TT warm, etc.).
    {
        let mut b = base.clone();
        let _ = sonetto.search_with_limits(&mut b, -INF, INF, depth, limits);
        let mut b = base.clone();
        let _ = sensei.best_move(&mut b, depth, limits);
        let _ = deriv.evaluate_backend(&mut sensei, &base, limits);
    }

    // Timed runs.
    println!("\n-- Budgeted (max_nodes={budget}, depth={depth}) --");

    bench_sonetto(&mut sonetto, &base, depth, limits);
    bench_sensei_ab(&mut sensei, &base, depth, limits);
    bench_derivative_sensei(&mut deriv, &mut sensei, &base, limits);

    // Also show a quick fixed-depth throughput sample (no budget) for AB engines.
    println!("\n-- Fixed-depth (no budget, depth={depth}) --");
    bench_sonetto(&mut sonetto, &base, depth, SearchLimits::default());
    bench_sensei_ab(&mut sensei, &base, depth, SearchLimits::default());
}

fn bench_sonetto(searcher: &mut Searcher, base: &Board, depth: u8, limits: SearchLimits) {
    let mut b = base.clone();
    let start = Instant::now();
    let out = searcher.search_with_limits(&mut b, -INF, INF, depth, limits);
    let dt = start.elapsed();

    let nps = nps(out.nodes, dt);

    println!(
        "Sonetto        | nodes {:>10} | {:>8.2} ms | {:>10.2} kn/s | aborted {} | mv {:>2} | score {}",
        out.nodes,
        ms(dt),
        nps / 1_000.0,
        out.aborted,
        out.best_move,
        out.score,
    );
}

fn bench_sensei_ab(sensei: &mut SenseiAlphaBeta, base: &Board, depth: u8, limits: SearchLimits) {
    let mut b = base.clone();
    let start = Instant::now();
    let (mv, score) = sensei.best_move(&mut b, depth, limits);
    let dt = start.elapsed();

    let nodes = sensei.last_nodes();
    let nps = nps(nodes, dt);

    println!(
        "SenseiAB       | nodes {:>10} | {:>8.2} ms | {:>10.2} kn/s | aborted {} | mv {:>2} | score {}",
        nodes,
        ms(dt),
        nps / 1_000.0,
        sensei.last_aborted(),
        mv,
        score,
    );
}

fn bench_derivative_sensei(
    deriv: &mut DerivativeEvaluator,
    sensei: &mut SenseiAlphaBeta,
    base: &Board,
    limits: SearchLimits,
) {
    let start = Instant::now();
    let res = deriv.evaluate_backend(sensei, base, limits);
    let dt = start.elapsed();

    let nps = nps(res.nodes_used, dt);

    // Top-N estimates come from the derivative arena state.
    let top = deriv.root_top_n_estimates(5);
    let best = top.get(0).copied().unwrap_or((PASS, 0));

    println!(
        "Derivative+AB  | nodes {:>10} | {:>8.2} ms | {:>10.2} kn/s | iters {:>5} | mv {:>2} | est {} | [L,U]=[{},{}]",
        res.nodes_used,
        ms(dt),
        nps / 1_000.0,
        res.iterations,
        best.0,
        best.1,
        res.lower,
        res.upper,
    );
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1000.0)
}

fn nps(nodes: u64, d: Duration) -> f64 {
    let s = d.as_secs_f64();
    if s <= 0.0 {
        0.0
    } else {
        nodes as f64 / s
    }
}

fn try_load_weights() -> Weights {
    // Default = all zeros (still functional).
    let mut weights = Weights::zeroed();

    if let Ok(bytes) = std::fs::read("crates/sonetto_core/data/eval.egev2.gz") {
        if let Ok(w) = decode_egev2(&bytes) {
            weights = w;
        } else {
            eprintln!("warning: failed to decode eval.egev2.gz; using zeroed weights");
        }
    } else {
        eprintln!("warning: missing eval.egev2.gz; using zeroed weights");
    }

    weights
}

/// Generate a deterministic-ish midgame board by alternating legal moves.
fn random_midgame_board(plies: usize) -> Board {
    let mut b = Board::new_start(sonetto_core::egev2::N_PATTERN_FEATURES);

    // Simple LCG for reproducible pseudo-random move selection.
    let mut seed: u64 = 0x1234_5678_9abc_def0;

    for _ in 0..plies {
        let mask = legal_moves(b.player, b.opponent);
        if mask == 0 {
            // Pass.
            let ok = b.apply_move(PASS);
            if !ok {
                break;
            }
            continue;
        }

        let mut moves = [PASS; 64];
        let n = push_moves_from_mask(mask, &mut moves);
        if n == 0 {
            break;
        }

        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (seed as usize) % n;
        let mv = moves[idx];

        let ok = b.apply_move(mv);
        if !ok {
            break;
        }
    }

    b
}
