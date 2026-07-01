//! Value-semantic tests for DMAS. Expected values are the closed-form pairwise
//! products ½[(Σŝ)² − Σŝ²], cross-checked against the explicit Σ_{i<j} ŝᵢŝⱼ.

use super::{delay_and_sum_dmas, dmas_combine};
use crate::signal_processing::beamforming::time_domain::delay_reference::DelayReference;
use ndarray::Array3;

#[test]
fn dmas_combine_two_channels_equals_their_product() {
    // ŝ = [2, 3] from x = [4, 9]; y = ½(5² − 13) = 6 = 2·3.
    assert!((dmas_combine(&[4.0, 9.0]) - 6.0).abs() < 1e-12);
    // x = [1, 1] ⇒ ŝ = [1,1] ⇒ y = ½(4 − 2) = 1 = 1·1.
    assert!((dmas_combine(&[1.0, 1.0]) - 1.0).abs() < 1e-12);
}

#[test]
fn dmas_combine_anti_phase_pair_is_negative() {
    // x = [1, -1] ⇒ ŝ = [1, -1] ⇒ y = ½(0 − 2) = −1 = ŝ₁ŝ₂ (sidelobe suppression).
    assert!((dmas_combine(&[1.0, -1.0]) - (-1.0)).abs() < 1e-12);
}

#[test]
fn dmas_combine_three_coherent_channels_matches_pair_count() {
    // x = [1,1,1] ⇒ ŝ=[1,1,1] ⇒ y = ½(9 − 3) = 3 = pairs {12,13,23}.
    assert!((dmas_combine(&[1.0, 1.0, 1.0]) - 3.0).abs() < 1e-12);
}

#[test]
fn dmas_combine_single_or_empty_has_no_pairs() {
    // Single channel: ½(ŝ² − ŝ²) = 0 analytically; the FP cancellation residual
    // is bounded by machine-ε·ŝ² (ŝ² = 5 here ⇒ ~1e-15).
    assert!(dmas_combine(&[5.0]).abs() < 1e-14);
    // Empty aperture: no terms ⇒ exactly 0.
    assert_eq!(dmas_combine(&[]), 0.0);
}

#[test]
fn dmas_combine_zero_sample_contributes_nothing() {
    // √0 = 0, so a zero channel drops out regardless of signum(0) convention:
    // x = [4, 0, 9] ⇒ same as [4, 9] ⇒ 6.
    assert!((dmas_combine(&[4.0, 0.0, 9.0]) - 6.0).abs() < 1e-12);
}

#[test]
fn dmas_combine_matches_explicit_pairwise_sum() {
    let x: [f64; 4] = [0.5, -2.0, 3.0, 1.5];
    let roots: Vec<f64> = x.iter().map(|&v| v.signum() * v.abs().sqrt()).collect();
    let mut expected = 0.0;
    for i in 0..roots.len() {
        for j in (i + 1)..roots.len() {
            expected += roots[i] * roots[j];
        }
    }
    assert!((dmas_combine(&x) - expected).abs() < 1e-12);
}

#[test]
fn active_dmas_preserves_aligned_coherent_impulse() {
    // Two aligned impulses [1,1] at the focus ⇒ DMAS = ½(2² − 2) = 1
    // (vs DAS = 2): DMAS reports the pairwise-correlated amplitude.
    let fs = 10.0;
    let n = 8usize;
    let delays = [1.0, 1.2];
    let weights = [1.0, 1.0];
    let mut x = Array3::<f64>::zeros((2, 1, n));
    x[[0, 0, 3]] = 1.0;
    x[[1, 0, 5]] = 1.0;

    let out = delay_and_sum_dmas(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect("dmas");
    assert!(
        (out[[0, 0, 3]] - 1.0).abs() < 1e-12,
        "focus DMAS should be 1, got {}",
        out[[0, 0, 3]]
    );
}
