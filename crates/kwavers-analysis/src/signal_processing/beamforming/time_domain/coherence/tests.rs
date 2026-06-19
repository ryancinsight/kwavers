//! Value-semantic tests for coherence-factor weighting.
//!
//! Expected values are derived analytically from the closed-form definitions in
//! [`super`]; no tolerance is wider than the f64 round-off of the arithmetic.

use super::{delay_and_sum_coherence, CoherenceFactor};
use crate::signal_processing::beamforming::time_domain::das::delay_and_sum;
use crate::signal_processing::beamforming::time_domain::delay_reference::DelayReference;
use ndarray::{Array2, Array3};

/// Build a single-sample aligned aperture column `(n_elements, 1)`.
fn column(values: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((values.len(), 1), values.to_vec()).expect("valid column shape")
}

// --- Amplitude coherence factor (Mallart & Fink 1994) ---

#[test]
fn amplitude_cf_is_unity_for_a_perfectly_coherent_aperture() {
    // All elements equal ⇒ CF = |N·a|² / (N · N·a²) = 1, independent of a.
    let aligned = column(&[2.0, 2.0, 2.0, 2.0]);
    let cf = CoherenceFactor::Amplitude.weights(&aligned).expect("weights");
    assert!((cf[0] - 1.0).abs() < 1e-12, "coherent CF should be 1, got {}", cf[0]);
}

#[test]
fn amplitude_cf_is_zero_for_a_zero_mean_aperture() {
    // Σxᵢ = 0 ⇒ coherent energy 0 ⇒ CF = 0.
    let aligned = column(&[1.0, -1.0, 1.0, -1.0]);
    let cf = CoherenceFactor::Amplitude.weights(&aligned).expect("weights");
    assert!(cf[0].abs() < 1e-12, "incoherent CF should be 0, got {}", cf[0]);
}

#[test]
fn amplitude_cf_matches_closed_form_intermediate() {
    // column [1,1,1,0]: Σx = 3, Σx² = 3, N = 4 ⇒ CF = 9 / (4·3) = 0.75.
    let aligned = column(&[1.0, 1.0, 1.0, 0.0]);
    let cf = CoherenceFactor::Amplitude.weights(&aligned).expect("weights");
    assert!((cf[0] - 0.75).abs() < 1e-12, "expected 0.75, got {}", cf[0]);
}

#[test]
fn amplitude_cf_is_zero_for_all_zero_column() {
    let aligned = column(&[0.0, 0.0, 0.0]);
    let cf = CoherenceFactor::Amplitude.weights(&aligned).expect("weights");
    assert_eq!(cf[0], 0.0);
}

#[test]
fn amplitude_cf_stays_in_unit_interval() {
    // Cauchy–Schwarz bound: CF ∈ [0,1] for arbitrary data.
    let aligned = Array2::from_shape_vec(
        (4, 3),
        vec![
            3.0, -1.0, 0.5, //
            -2.0, 4.0, -7.0, //
            1.5, 0.0, 2.0, //
            -0.25, 9.0, -3.0,
        ],
    )
    .expect("valid shape");
    let cf = CoherenceFactor::Amplitude.weights(&aligned).expect("weights");
    for &v in cf.iter() {
        assert!((0.0..=1.0).contains(&v), "CF out of [0,1]: {v}");
    }
}

// --- Sign coherence factor (Camacho et al. 2009) ---

#[test]
fn sign_cf_is_unity_for_uniform_sign() {
    // All positive ⇒ b̄ = 1 ⇒ SCF = (1 − 0)^p = 1.
    let aligned = column(&[0.5, 2.0, 0.1, 3.0]);
    let cf = CoherenceFactor::Sign { sensitivity: 1.0 }
        .weights(&aligned)
        .expect("weights");
    assert!((cf[0] - 1.0).abs() < 1e-12, "uniform-sign SCF should be 1, got {}", cf[0]);
}

#[test]
fn sign_cf_is_zero_for_balanced_signs() {
    // Equal +/- counts ⇒ b̄ = 0 ⇒ SCF = (1 − 1)^p = 0.
    let aligned = column(&[1.0, -1.0, 2.0, -2.0]);
    let cf = CoherenceFactor::Sign { sensitivity: 1.0 }
        .weights(&aligned)
        .expect("weights");
    assert!(cf[0].abs() < 1e-12, "balanced-sign SCF should be 0, got {}", cf[0]);
}

#[test]
fn sign_cf_matches_closed_form_intermediate() {
    // 3 positive, 1 negative, N=4 ⇒ b̄ = 0.5 ⇒ SCF = 1 − √(1 − 0.25) = 1 − √0.75.
    let aligned = column(&[1.0, 1.0, 1.0, -1.0]);
    let expected = 1.0 - 0.75_f64.sqrt();
    let cf = CoherenceFactor::Sign { sensitivity: 1.0 }
        .weights(&aligned)
        .expect("weights");
    assert!((cf[0] - expected).abs() < 1e-12, "expected {expected}, got {}", cf[0]);
}

#[test]
fn sign_cf_sensitivity_exponentiates() {
    // Same b̄ = 0.5; sensitivity p=2 ⇒ SCF = (1 − √0.75)².
    let aligned = column(&[1.0, 1.0, 1.0, -1.0]);
    let base = 1.0 - 0.75_f64.sqrt();
    let cf = CoherenceFactor::Sign { sensitivity: 2.0 }
        .weights(&aligned)
        .expect("weights");
    assert!((cf[0] - base * base).abs() < 1e-12, "expected {}, got {}", base * base, cf[0]);
}

#[test]
fn sign_cf_rejects_invalid_sensitivity() {
    let aligned = column(&[1.0, 1.0]);
    assert!(CoherenceFactor::Sign { sensitivity: 0.5 }.weights(&aligned).is_err());
    assert!(CoherenceFactor::Sign { sensitivity: f64::NAN }.weights(&aligned).is_err());
}

// --- End-to-end DAS + coherence ---

#[test]
fn coherence_das_equals_das_times_cf() {
    // Zero delays ⇒ aligned = raw samples; verify y_cf[j] == cf[j]·y_das[j].
    let fs = 1.0;
    let delays = [0.0, 0.0, 0.0];
    let weights = [1.0, 0.5, 2.0];
    let data = Array3::from_shape_vec(
        (3, 1, 4),
        vec![
            1.0, -2.0, 3.0, 0.0, //
            0.5, 2.0, -1.0, 4.0, //
            -1.0, 1.0, 2.0, -2.0,
        ],
    )
    .expect("valid shape");

    let das = delay_and_sum(&data, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect("das");
    let (cf_out, cf) = delay_and_sum_coherence(
        &data,
        fs,
        &delays,
        &weights,
        DelayReference::SensorIndex(0),
        CoherenceFactor::Amplitude,
    )
    .expect("coherence das");

    for j in 0..4 {
        let expected = das[[0, 0, j]] * cf[j];
        assert!(
            (cf_out[[0, 0, j]] - expected).abs() < 1e-12,
            "j={j}: expected {expected}, got {}",
            cf_out[[0, 0, j]]
        );
    }
}

#[test]
fn coherence_preserves_aligned_coherent_impulse() {
    // Two aligned impulses: at the focus column = [1,1] ⇒ CF = 1 ⇒ output preserved.
    let fs = 10.0;
    let n = 8usize;
    let delays = [1.0, 1.2];
    let weights = [1.0, 1.0];
    let mut x = Array3::<f64>::zeros((2, 1, n));
    x[[0, 0, 3]] = 1.0;
    x[[1, 0, 5]] = 1.0;

    let (out, cf) = delay_and_sum_coherence(
        &x,
        fs,
        &delays,
        &weights,
        DelayReference::SensorIndex(0),
        CoherenceFactor::Amplitude,
    )
    .expect("coherence das");

    assert!((cf[3] - 1.0).abs() < 1e-12, "focus CF should be 1, got {}", cf[3]);
    assert!((out[[0, 0, 3]] - 2.0).abs() < 1e-12, "focus output should be 2, got {}", out[[0, 0, 3]]);
}
