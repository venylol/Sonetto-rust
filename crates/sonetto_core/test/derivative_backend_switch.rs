use sonetto_core::board::{Board, Color, Undo};
use sonetto_core::coord::PASS;
use sonetto_core::derivative::{DerivativeConfig, DerivativeEvaluator};
use sonetto_core::eval::{build_sonetto_feature_defs_and_occ, Weights, N_PATTERN_FEATURES};
use sonetto_core::features::swap::build_swap_tables;
use sonetto_core::movegen::{legal_moves, push_moves_from_mask};
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

#[test]
fn derivative_backend_switch() {
    let (feats, occ) = build_sonetto_feature_defs_and_occ();
    let swap = build_swap_tables(&feats);

    let mut searcher = Searcher::new(1, Weights::zeroed(), feats.clone(), swap.clone(), occ.clone());
    let mut ab = SenseiAlphaBeta::new(Weights::zeroed(), feats.clone(), swap.clone(), occ.clone());

    // Use a near-end position so the Derivative scheduler may exercise proof search
    // (i.e., it will call `exact_search_with_limits` on the backend).
    let mut board = Board::new_start(Color::Black, N_PATTERN_FEATURES);
    random_play(&mut board, &mut Rng(0xC0FFEE), 54);

    let mut cfg = DerivativeConfig::default().with_tree_node_cap(20_000);
    cfg.seed_max_nodes = 1_000;
    // Stage 5 uses `solve_min_nodes..solve_max_nodes` for proof attempts.
    cfg.solve_min_nodes = 1_000;
    cfg.solve_max_nodes = 10_000;
    cfg.solve_max_empties = 12;

    let mut evaluator = DerivativeEvaluator::new(cfg);

    let limits = SearchLimits {
        max_nodes: Some(30_000),
    };

    // Switch backend: Searcher -> SenseiAlphaBeta -> Searcher
    let r1 = evaluator.evaluate(&mut searcher, &board, limits);
    assert_legal_move(&board, r1.best_move);

    let r2 = evaluator.evaluate_backend(&mut ab, &board, limits);
    assert_legal_move(&board, r2.best_move);

    let r3 = evaluator.evaluate(&mut searcher, &board, limits);
    assert_legal_move(&board, r3.best_move);
}
