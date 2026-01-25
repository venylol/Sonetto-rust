//! Simple nodes/sec comparison between the default Searcher and SenseiAlphaBeta.
//!
//! Run (from repo root):
//! ```bash
//! cargo run -p sonetto_core --example sensei_ab_bench --release
//! ```
//!
//! By default this uses `Weights::zeroed()` (fast, deterministic, no large
//! allocations). If you want to run with the repository's `eval.egev2.gz`,
//! set `SONETTO_EGEV2_GZ` to point at the file and ensure `gzip` is available:
//! ```bash
//! SONETTO_EGEV2_GZ=eval.egev2.gz cargo run -p sonetto_core --example sensei_ab_bench --release
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use sonetto_core::board::Board;
use sonetto_core::eval::{build_sonetto_feature_defs_and_occ, decode_egev2, Weights, N_PATTERN_FEATURES};
use sonetto_core::features::swap::SwapTables;
use sonetto_core::movegen::{legal_moves, push_moves_from_mask};
use sonetto_core::search::{SearchLimits, Searcher};
use sonetto_core::sensei_ab::SenseiAlphaBeta;

const INF: i32 = 1_000_000;

#[derive(Clone, Copy)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn gen_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
}

fn find_eval_gz() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SONETTO_EGEV2_GZ") {
        return Some(PathBuf::from(p));
    }

    // Walk up from this crate's manifest dir.
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..8 {
        let cand = dir.join("eval.egev2.gz");
        if cand.exists() {
            return Some(cand);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn try_load_weights() -> Weights {
    let Some(path) = find_eval_gz() else {
        eprintln!("[bench] SONETTO_EGEV2_GZ not set and eval.egev2.gz not found; using zeroed weights");
        return Weights::zeroed();
    };

    // Use external gzip to keep the example dependency-free.
    // If gzip is unavailable or decoding fails, fall back to zeroed.
    match Command::new("gzip").arg("-cd").arg(&path).output() {
        Ok(out) if out.status.success() => match decode_egev2(&out.stdout) {
            Ok(w) => {
                eprintln!("[bench] loaded weights from {}", path.display());
                w
            }
            Err(e) => {
                eprintln!("[bench] decode_egev2 failed ({e}); using zeroed weights");
                Weights::zeroed()
            }
        },
        Ok(out) => {
            eprintln!("[bench] gzip failed (status={}); using zeroed weights", out.status);
            Weights::zeroed()
        }
        Err(e) => {
            eprintln!("[bench] failed to spawn gzip ({e}); using zeroed weights");
            Weights::zeroed()
        }
    }
}

fn random_midgame_board(mut rng: Rng, plies: usize) -> Board {
    let mut board = Board::new_start(N_PATTERN_FEATURES);

    let mut undo = Default::default();
    for _ in 0..plies {
        let mask = legal_moves(board.player, board.opponent);
        if mask == 0 {
            let opp_mask = legal_moves(board.opponent, board.player);
            if opp_mask == 0 {
                break;
            }
            board.apply_move(PASS, &mut undo);
            continue;
        }

        let mut moves = [0u8; 64];
        let n = push_moves_from_mask(mask, &mut moves);
        let mv = moves[rng.gen_usize(n)];
        board.apply_move(mv, &mut undo);
    }

    board
}

use sonetto_core::coord::PASS;

fn main() {
    let weights = try_load_weights();

    // Build feature defs and the occupancy map.
    let (feats_a, occ_a) = build_sonetto_feature_defs_and_occ();
    let swap_a = SwapTables::build_swap_tables();

    let (feats_b, occ_b) = build_sonetto_feature_defs_and_occ();
    let swap_b = SwapTables::build_swap_tables();

    let mut searcher = Searcher::new(16, weights.clone(), feats_a, swap_a, occ_a);
    let mut sensei = SenseiAlphaBeta::new(weights, feats_b, swap_b, occ_b);

    let depth: u8 = 6;
    let board = random_midgame_board(Rng::new(0xC0FFEE), 12);

    println!("Board empties: {}", board.empty_count);
    println!("Depth: {}", depth);

    // Searcher
    let mut board_a = board.clone();
    let t0 = Instant::now();
    let _outcome = searcher.search_with_limits(&mut board_a, -INF, INF, depth, SearchLimits { max_nodes: None });
    let dt = t0.elapsed().as_secs_f64().max(1e-9);
    let nodes = searcher.last_nodes();
    println!("Searcher: nodes={} time={:.4}s nodes/sec={:.0}", nodes, dt, (nodes as f64) / dt);

    // SenseiAlphaBeta
    let mut board_b = board.clone();
    let t1 = Instant::now();
    let (_mv, _score) = sensei.best_move(&mut board_b, depth, SearchLimits { max_nodes: None });
    let dt2 = t1.elapsed().as_secs_f64().max(1e-9);
    let nodes2 = sensei.last_nodes();
    println!("SenseiAlphaBeta: nodes={} time={:.4}s nodes/sec={:.0}", nodes2, dt2, (nodes2 as f64) / dt2);
}
