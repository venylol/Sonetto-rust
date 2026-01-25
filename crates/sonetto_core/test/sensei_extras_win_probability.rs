use sonetto_core::sensei_extras::win_probability::{
    byte_to_probability, byte_to_probability_explicit, cdf_offset_to_depth_empties_eval,
    data_to_cdf_offset, gaussian_cdf, probability_to_byte_explicit, MAX_CDF_OFFSET, PROB_STEP,
};

#[test]
fn gaussian_cdf_matches_known_values_and_symmetry() {
    // Wikipedia standard normal table reference point.
    let v = gaussian_cdf(-1.0, 0.0, 1.0);
    assert!((v - 1.58655e-1).abs() < 1e-4, "v={v}");

    // Symmetry: Φ(x) = 1 - Φ(-x).
    for i in 0..=100 {
        let x = (i as f64) / 10.0;
        let a = gaussian_cdf(x, 0.0, 1.0);
        let b = 1.0 - gaussian_cdf(-x, 0.0, 1.0);
        assert!((a - b).abs() < 1e-8, "x={x} a={a} b={b}");
    }
}

#[test]
fn probability_byte_roundtrip_matches_sensei_contract() {
    assert_eq!(probability_to_byte_explicit(0.0), 0);
    assert_eq!(probability_to_byte_explicit(1.0), PROB_STEP);

    // A low byte should map to a small probability.
    let p10 = byte_to_probability_explicit(10);
    assert!((p10 - 0.0).abs() < 0.1, "p10={p10}");

    // Table roundtrip: ProbabilityToByteExplicit(ByteToProbability(i)) == i.
    for i in 0u8..=PROB_STEP {
        let p = byte_to_probability(i);
        let b = probability_to_byte_explicit(p);
        assert_eq!(b, i, "i={i} p={p} b={b}");
    }
}

#[test]
fn delta_to_cdf_offset_roundtrip_full_coverage() {
    // Matches the original Sensei test: ensure the bit packing is consistent.
    for depth in 1u8..=4 {
        for delta in (-128 * 8)..=(128 * 8) {
            for empties in 0u8..64 {
                let off = data_to_cdf_offset(depth, empties, delta);
                assert!(off < MAX_CDF_OFFSET, "off={off} depth={depth} empties={empties} delta={delta}");

                let (d2, e2, de2) = cdf_offset_to_depth_empties_eval(off);
                assert_eq!(d2, depth);
                assert_eq!(e2, empties);
                assert_eq!(de2, delta);
            }
        }
    }
}
