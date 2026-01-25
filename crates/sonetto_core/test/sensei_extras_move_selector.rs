use sonetto_core::sensei_extras::move_selector::{
    find_next_move_index_with_rng, get_move_multiplier, Rng64, XorShift64,
};

#[test]
fn get_move_multiplier_sums_to_two() {
    let mut sum = 0.0f64;
    for mv in 0..60usize {
        sum += get_move_multiplier(mv);
    }
    assert!((sum - 2.0).abs() < 1e-12, "sum={sum}");
}

#[test]
fn find_next_move_empty_returns_none() {
    let evals: [f64; 0] = [];
    let mut rng = XorShift64::new(1);
    assert_eq!(find_next_move_index_with_rng(&evals, 1.0, 100.0, &mut rng), None);
}

fn run_two_moves(expected_error: f64) {
    // Two moves: best eval=0, worse eval=6 => error is either 0 or 6.
    let evals = [0.0f64, 6.0f64];
    let mut rng = XorShift64::new(0xC0FFEE_u64);

    let n = 10_000usize;
    let mut total_err = 0.0f64;

    for _ in 0..n {
        let idx = find_next_move_index_with_rng(&evals, expected_error, 1e9, &mut rng)
            .expect("non-empty eval list");
        if idx == 1 {
            total_err += 6.0;
        }
    }

    let avg = total_err / (n as f64);
    let expected = expected_error.min(3.0);
    assert!(
        (avg - expected).abs() < 0.05,
        "expected_error={expected_error} avg={avg} expected={expected}"
    );
}

fn run_three_moves(expected_error: f64) {
    // Three moves: evals 0, 3, 6 => errors 0,3,6.
    let evals = [0.0f64, 3.0f64, 6.0f64];
    let mut rng = XorShift64::new(0xDEAD_BEEF_u64);

    let n = 20_000usize;
    let mut count = [0usize; 3];

    for _ in 0..n {
        let idx = find_next_move_index_with_rng(&evals, expected_error, 1e9, &mut rng)
            .expect("non-empty eval list");
        count[idx] += 1;
    }

    let avg = (count[1] as f64 * 3.0 + count[2] as f64 * 6.0) / (n as f64);
    let expected = expected_error.min(3.0);
    assert!(
        (avg - expected).abs() < 0.05,
        "expected_error={expected_error} avg={avg} expected={expected} counts={count:?}"
    );

    if expected_error > 0.0 {
        // With exp(-lambda*err), ratios for steps of 3 should be approximately equal:
        // p(err=0)/p(err=3) ≈ p(err=3)/p(err=6).
        // Allow a fairly loose tolerance to keep the test stable.
        assert!(count[1] > 0 && count[2] > 0, "counts={count:?}");
        let ratio1 = count[0] as f64 / count[1] as f64;
        let ratio2 = count[1] as f64 / count[2] as f64;
        assert!(
            (ratio1 - ratio2).abs() < 0.15,
            "expected_error={expected_error} ratio1={ratio1} ratio2={ratio2} counts={count:?}"
        );
    }
}

#[test]
fn find_next_move_two_moves_matches_expected_error_schedule() {
    for expected_error in [0.0f64, 1.0, 2.0, 3.0, 6.0] {
        run_two_moves(expected_error);
    }
}

#[test]
fn find_next_move_three_moves_matches_expected_error_and_ratios() {
    for expected_error in [0.0f64, 1.0, 2.0, 3.0, 6.0] {
        run_three_moves(expected_error);
    }
}

#[test]
fn find_next_move_respects_max_error_exclusion() {
    // Third move is slightly worse than max_error so it should be excluded.
    let evals = [0.0f64, 6.0f64, 7.0001f64];
    let mut rng = XorShift64::new(0xABCD_EF01_u64);

    for expected_error in [0.0f64, 1.0, 2.0, 3.0, 6.0] {
        for _ in 0..5_000 {
            let idx = find_next_move_index_with_rng(&evals, expected_error, 7.0, &mut rng)
                .expect("non-empty eval list");
            assert_ne!(idx, 2, "max_error should exclude idx=2");
        }
    }
}

// Compile-time sanity: ensure our RNG satisfies the trait.
fn _assert_rng64_impl<T: Rng64>() {}

#[test]
fn xorshift64_implements_rng64() {
    _assert_rng64_impl::<XorShift64>();
}
