use sonetto_core::board::{Board, Color, Undo};
use sonetto_core::coord::PASS;
use sonetto_core::derivative::{DerivativeConfig, DerivativeEvaluator};
use sonetto_core::eval::{build_sonetto_feature_defs_and_occ, Weights, N_PATTERN_FEATURES};
use sonetto_core::features::swap::build_swap_tables;
use sonetto_core::movegen::{legal_moves, push_moves_from_mask};
use sonetto_core::score::{MAX_DISC_DIFF, SCALE};
use sonetto_core::search::{SearchLimits, Searcher};
use sonetto_core::sensei_ab::SenseiAlphaBeta;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn random_play(board: &mut Board, rng: &mut Rng, n_plies: u8) {
    for _ in 0..n_plies {
        let mask = legal_moves(board.player, board.opponent);
        let mv = if mask == 0 {
            if legal_moves(board.opponent, board.player) == 0 {
                break;
            }
            PASS
        } else {
            let mut moves = [PASS; 64];
            let n = push_moves_from_mask(mask, &mut moves);
            moves[(rng.next_u64() as usize) % n]
        };

        let mut undo = Undo::default();
        board.apply_move(mv, &mut undo);
    }
}

fn assert_legal_move(board: &Board, mv: u8) {
    let mask = legal_moves(board.player, board.opponent);
    if mask == 0 {
        assert_eq!(mv, PASS, "expected PASS when no legal moves exist");
    } else {
        assert_ne!(mv, PASS, "unexpected PASS when legal moves exist");
        assert!(
            ((mask >> mv) & 1) != 0,
            "move {mv} is not legal in this position"
        );
    }
}

fn score_min() -> i32 {
    -(MAX_DISC_DIFF as i32) * SCALE
}

fn score_max() -> i32 {
    (MAX_DISC_DIFF as i32) * SCALE
}


fn mk_cfg() -> DerivativeConfig {
    let mut cfg = DerivativeConfig::default().with_tree_node_cap(12_000);
    cfg.seed_max_nodes = 1_000;
    cfg.solve_min_nodes = 1_000;
    cfg.solve_max_nodes = 4_000;
    cfg
}

#[test]
fn derivative_bounds_monotonic() {
    let (feats, occ) = build_sonetto_feature_defs_and_occ();
    let swap = build_swap_tables(&feats);

    let mut board = Board::new_start(Color::Black, N_PATTERN_FEATURES);
    random_play(&mut board, &mut Rng(0x1234_5678), 26);

    let limits = SearchLimits {
        max_nodes: Some(12_000),
    };

    // --- Searcher backend ---
    let mut searcher = Searcher::new(1, Weights::zeroed(), feats.clone(), swap.clone(), occ.clone());
    let mut evaluator = DerivativeEvaluator::new(mk_cfg());

    let r1 = evaluator.evaluate_with_bounds(&mut searcher, &board, score_min(), score_max(), limits);
    assert!(r1.lower <= r1.upper, "bounds must remain consistent");
    assert!(r1.lower >= score_min(), "lower must not decrease");
    assert!(r1.upper <= score_max(), "upper must not increase");
    assert_legal_move(&board, r1.best_move);

    let r2 = evaluator.evaluate_with_bounds(&mut searcher, &board, r1.lower, r1.upper, limits);
    assert!(r2.lower <= r2.upper, "bounds must remain consistent");
    assert!(
        r2.lower >= r1.lower,
        "lower must be monotone non-decreasing across calls"
    );
    assert!(
        r2.upper <= r1.upper,
        "upper must be monotone non-increasing across calls"
    );
    assert_legal_move(&board, r2.best_move);

    // --- SenseiAlphaBeta backend ---
    let mut ab = SenseiAlphaBeta::new(Weights::zeroed(), feats.clone(), swap.clone(), occ.clone());
    let mut evaluator2 = DerivativeEvaluator::new(mk_cfg());

    let s1 = evaluator2.evaluate_with_bounds_backend(&mut ab, &board, score_min(), score_max(), limits);
    assert!(s1.lower <= s1.upper, "bounds must remain consistent");
    assert!(s1.lower >= score_min(), "lower must not decrease");
    assert!(s1.upper <= score_max(), "upper must not increase");
    assert_legal_move(&board, s1.best_move);

    let s2 = evaluator2.evaluate_with_bounds_backend(&mut ab, &board, s1.lower, s1.upper, limits);
    assert!(s2.lower <= s2.upper, "bounds must remain consistent");
    assert!(
        s2.lower >= s1.lower,
        "lower must be monotone non-decreasing across calls"
    );
    assert!(
        s2.upper <= s1.upper,
        "upper must be monotone non-increasing across calls"
    );
    assert_legal_move(&board, s2.best_move);
}
