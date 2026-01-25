use std::fs;
use std::path::PathBuf;

use sonetto_core::board::{Board, Color};
use sonetto_core::coord::PASS;
use sonetto_core::eval::{build_sonetto_feature_defs_and_occ, Weights, N_PATTERN_FEATURES, N_PHASES};
use sonetto_core::features::swap::SwapTables;
use sonetto_core::search::{AnalyzeMode, AnalyzeTopNRequest, SearchLimits, Searcher};
use sonetto_core::sensei_ab::SenseiAlphaBeta;
use sonetto_core::zobrist::compute_hash;

/// Deterministic xorshift RNG.
#[derive(Clone, Copy, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state.
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_i16(&mut self) -> i16 {
        // Small range to keep eval sums tame.
        let v = (self.next_u64() % 41) as i16; // 0..=40
        v - 20 // -20..=20
    }
}

fn load_test_positions() -> Vec<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("testdata");
    path.push("sensei_ab_regression.txt");

    let data = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    data.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect()
}

fn parse_board_str(s: &str) -> Board {
    assert_eq!(s.len(), 65, "board strings must be 64 cells + side char");

    let bytes = s.as_bytes();
    let mut black = 0u64;
    let mut white = 0u64;

    for i in 0..64 {
        match bytes[i] {
            b'X' => black |= 1u64 << i,
            b'O' => white |= 1u64 << i,
            b'-' => {}
            other => panic!("unexpected board char {other} at idx {i}"),
        }
    }

    let side = match bytes[64] {
        b'X' => Color::Black,
        b'O' => Color::White,
        other => panic!("unexpected side char {other}"),
    };

    let (player, opponent) = match side {
        Color::Black => (black, white),
        Color::White => (white, black),
    };

    let occupied = black | white;
    let empty_count = 64 - occupied.count_ones() as u8;

    let mut board = Board::new_empty(side, N_PATTERN_FEATURES);
    board.player = player;
    board.opponent = opponent;
    board.empty_count = empty_count;
    board.hash = compute_hash(board.bits_by_color(), board.side);

    // Features are recomputed by both backends.
    board.feat_is_pattern_ids = false;

    board
}

fn make_test_weights(phases: usize) -> Weights {
    assert!(phases > 0);

    let params_per_phase = Weights::expected_len() / N_PHASES;
    let len = params_per_phase * phases;

    let mut rng = XorShift64::new(0xC0FFEE);
    let mut params = Vec::with_capacity(len);
    for _ in 0..len {
        params.push(rng.next_i16());
    }

    Weights::from_vec(params)
}

#[test]
fn sensei_ab_matches_sonetto_depth1_on_testdata() {
    let positions = load_test_positions();
    assert!(!positions.is_empty(), "testdata must not be empty");

    // Keep this reasonably small so CI doesn't explode.
    // The testdata is generated from early-game positions.
    let weights = make_test_weights(16);

    let (feats_a, occ_a) = build_sonetto_feature_defs_and_occ();
    let swap_a = SwapTables::build_swap_tables();
    let mut searcher = Searcher::new(1, weights.clone(), feats_a, swap_a, occ_a);

    let (feats_b, occ_b) = build_sonetto_feature_defs_and_occ();
    let swap_b = SwapTables::build_swap_tables();
    let mut sensei = SenseiAlphaBeta::new(weights, feats_b, swap_b, occ_b);

    let req = AnalyzeTopNRequest {
        mode: AnalyzeMode::Midgame { depth: 1 },
        limits: SearchLimits { max_nodes: None },
        top_n: 64,
    };

    for (idx, line) in positions.iter().enumerate() {
        let mut b0 = parse_board_str(line);
        let mut b1 = b0.clone();

        let r1 = searcher.analyze_top_n(&mut b0, req);
        let r2 = sensei.analyze_top_n(&mut b1, req);

        // If there are no legal moves, both should report no pairs.
        if r1.pairs.is_empty() {
            assert!(r2.pairs.is_empty(), "case {idx}: expected empty pairs");
            // Also sanity-check PASS handling.
            let (mv, _) = sensei.best_move(&mut b1, 1, SearchLimits { max_nodes: None });
            assert_eq!(mv, PASS, "case {idx}: expected PASS");
            continue;
        }

        let mut p1 = r1.pairs.clone();
        let mut p2 = r2.pairs.clone();
        p1.sort_by_key(|(mv, _)| *mv);
        p2.sort_by_key(|(mv, _)| *mv);

        assert_eq!(p1.len(), p2.len(), "case {idx}: move count mismatch");

        for ((mv1, sc1), (mv2, sc2)) in p1.iter().zip(p2.iter()) {
            assert_eq!(mv1, mv2, "case {idx}: move mismatch");
            assert_eq!(sc1, sc2, "case {idx}: score mismatch at mv {mv1}");
        }
    }
}
