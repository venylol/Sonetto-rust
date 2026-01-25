use sonetto_core::movegen::{legal_moves, push_moves_from_mask};

#[inline(always)]
fn has_bit(bb: u64, sq: u8) -> bool {
    ((bb >> (sq as u64)) & 1) != 0
}

/// Slow but obvious reference (scan64) for legal moves.
/// Works on internal bitpos row-major coordinates.
fn naive_legal_moves(me: u64, opp: u64) -> u64 {
    let occ = me | opp;
    let mut out = 0u64;

    const DIRS: [(i8, i8); 8] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ];

    for sq in 0u8..64 {
        if has_bit(occ, sq) {
            continue;
        }

        let r0 = (sq >> 3) as i8;
        let c0 = (sq & 7) as i8;

        let mut is_legal = false;
        for (dr, dc) in DIRS {
            let mut r = r0 + dr;
            let mut c = c0 + dc;
            let mut seen_opp = false;

            while r >= 0 && r < 8 && c >= 0 && c < 8 {
                let s = (r * 8 + c) as u8;
                if has_bit(opp, s) {
                    seen_opp = true;
                    r += dr;
                    c += dc;
                    continue;
                }
                if seen_opp && has_bit(me, s) {
                    is_legal = true;
                }
                break;
            }
            if is_legal {
                break;
            }
        }

        if is_legal {
            out |= 1u64 << (sq as u64);
        }
    }

    out
}

#[derive(Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

#[test]
fn legal_moves_initial_position_black_to_move() {
    // Standard start (internal bitpos row-major):
    // White at (3,3)=27 and (4,4)=36
    // Black at (3,4)=28 and (4,3)=35
    let me = (1u64 << 28) | (1u64 << 35);
    let opp = (1u64 << 27) | (1u64 << 36);

    let got = legal_moves(me, opp);

    // Expected legal moves for black:
    // (2,3)=19, (3,2)=26, (4,5)=37, (5,4)=44
    let expected = (1u64 << 19) | (1u64 << 26) | (1u64 << 37) | (1u64 << 44);
    assert_eq!(got, expected);

    // Cross-check vs slow reference.
    assert_eq!(got, naive_legal_moves(me, opp));
}

#[test]
fn legal_moves_matches_naive_on_random_positions() {
    let mut rng = XorShift64::new(0xC0FFEE12_3456_78u64);

    for _case in 0..5000 {
        let me = rng.next_u64();
        let mut opp = rng.next_u64();
        opp &= !me; // enforce disjoint

        let fast = legal_moves(me, opp);
        let slow = naive_legal_moves(me, opp);
        assert_eq!(fast, slow, "me={me:#018x} opp={opp:#018x}");
    }
}

#[test]
fn push_moves_from_mask_lists_all_bits_in_order() {
    let mut rng = XorShift64::new(0x1234_5678_9ABC_DEF0u64);

    for _case in 0..1000 {
        let mask = rng.next_u64();
        let mut out = [0u8; 64];
        let n = push_moves_from_mask(mask, &mut out);

        assert_eq!(n, mask.count_ones() as usize);

        // Ensure strictly increasing, and every output square is in `mask`.
        let mut prev: i32 = -1;
        for i in 0..n {
            let sq = out[i];
            assert!(has_bit(mask, sq));
            assert!((sq as i32) > prev);
            prev = sq as i32;
        }
    }
}
