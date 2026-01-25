use sonetto_core::board::Board;
use sonetto_core::coord::PASS;
use sonetto_core::eval::{build_sonetto_feature_defs_and_occ, Weights, N_PATTERN_FEATURES};
use sonetto_core::features::swap::SwapTables;
use sonetto_core::movegen::{legal_moves, push_moves_from_mask};
use sonetto_core::search::SearchLimits;
use sonetto_core::sensei_ab::SenseiAlphaBeta;

/// Tiny deterministic RNG (xorshift64*).
#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(2685821657736338717)
    }

    fn gen_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
}

fn random_play(mut rng: Rng, plies: usize) -> Board {
    let mut board = Board::new_start(N_PATTERN_FEATURES);

    for _ in 0..plies {
        let moves = legal_moves(board.player, board.opponent);
        if moves == 0 {
            // Pass or game over.
            let opp_moves = legal_moves(board.opponent, board.player);
            if opp_moves == 0 {
                break;
            }
            let mut undo = Default::default();
            board.apply_move(PASS, &mut undo);
            continue;
        }

        let mut list = [0u8; 64];
        let n = push_moves_from_mask(&mut list, moves);
        let mv = list[rng.gen_usize(n)];
        let mut undo = Default::default();
        board.apply_move(mv, &mut undo);
    }

    board
}

#[test]
fn sensei_ab_returns_legal_moves_on_random_positions() {
    let (feats, occ) = build_sonetto_feature_defs_and_occ();
    let swap = SwapTables::build_swap_tables();
    let weights = Weights::zeroed();

    let mut backend = SenseiAlphaBeta::new(weights, feats, swap, occ);

    let limits = SearchLimits { max_nodes: Some(200_000) };

    for i in 0..200u64 {
        let mut board = random_play(Rng::new(0xC0FFEE_u64 ^ i), 20);

        let before_player = board.player;
        let before_opponent = board.opponent;
        let before_side = board.side;

        let (mv, _score) = backend.best_move(&mut board, 4, limits);

        // Search should not change the position.
        assert_eq!(board.player, before_player);
        assert_eq!(board.opponent, before_opponent);
        assert_eq!(board.side, before_side);

        let moves = legal_moves(board.player, board.opponent);
        if moves == 0 {
            assert_eq!(mv, PASS);
        } else {
            let mv_bit = 1u64 << mv;
            assert_ne!(mv, PASS);
            assert_ne!(mv_bit & moves, 0, "returned move must be legal");
        }
    }
}
