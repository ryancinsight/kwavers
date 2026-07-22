use super::MinimumVariance;
use crate::signal_processing::beamforming::test_utilities;
use eunomia::assert_relative_eq;
use eunomia::Complex64;
use leto::{Array1, Array2};
use std::f64::consts::PI;

#[test]
fn mvdr_computes_finite_weights() {
    let n = 8;
    let cov = test_utilities::create_diagonal_dominant_covariance(n, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let mvdr = MinimumVariance::default();
    let weights = mvdr
        .compute_weights(&cov, &steering)
        .expect("weights should compute");

    assert_eq!(weights.len(), n);
    for &w in weights.iter() {
        assert!(
            w.re.is_finite() && w.im.is_finite(),
            "weight should be finite"
        );
    }
}

#[test]
fn mvdr_satisfies_unit_gain_constraint() {
    let n = 8;
    let cov = test_utilities::create_diagonal_dominant_covariance(n, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let mvdr = MinimumVariance::with_diagonal_loading(1e-4);
    let weights = mvdr
        .compute_weights(&cov, &steering)
        .expect("weights should compute");

    let gain: Complex64 = weights
        .iter()
        .zip(steering.iter())
        .map(|(&w, &a)| w.conj() * a)
        .sum();

    assert_relative_eq!(gain.re, 1.0, epsilon = 1e-6);
    assert_relative_eq!(gain.im, 0.0, epsilon = 1e-6);
}

#[test]
fn mvdr_with_different_angles() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);

    let mvdr = MinimumVariance::default();

    for angle_deg in [-30.0, 0.0, 30.0] {
        let angle_rad = angle_deg * PI / 180.0;
        let steering = test_utilities::create_steering_vector(n, angle_rad);

        let weights = mvdr
            .compute_weights(&cov, &steering)
            .expect("weights should compute");

        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(&w, &a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.norm(), 1.0, epsilon = 1e-6);
    }
}

#[test]
fn mvdr_pseudospectrum_is_positive() {
    let n = 8;
    let cov = test_utilities::create_diagonal_dominant_covariance(n, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let mvdr = MinimumVariance::with_diagonal_loading(1e-6);
    let spectrum = mvdr
        .pseudospectrum(&cov, &steering)
        .expect("pseudospectrum should compute");

    assert!(spectrum > 0.0);
    assert!(spectrum.is_finite());
}

#[test]
fn mvdr_rejects_complex_denominator_in_weights() {
    let cov = Array2::from_elem((1, 1), Complex64::new(1.0, -1.0));
    let steering = Array1::from_elem(1, Complex64::new(1.0, 0.0));

    let mvdr = MinimumVariance::new();
    let err = mvdr
        .compute_weights(&cov, &steering)
        .expect_err("non-Hermitian covariance should produce a complex denominator");

    assert!(
        err.to_string().contains("imaginary component"),
        "expected imaginary-denominator diagnostic, got {err}"
    );
}

#[test]
fn mvdr_rejects_complex_denominator_in_pseudospectrum() {
    let cov = Array2::from_elem((1, 1), Complex64::new(1.0, -1.0));
    let steering = Array1::from_elem(1, Complex64::new(1.0, 0.0));

    let mvdr = MinimumVariance::new();
    let err = mvdr
        .pseudospectrum(&cov, &steering)
        .expect_err("non-Hermitian covariance should produce a complex denominator");

    assert!(
        err.to_string().contains("imaginary component"),
        "expected imaginary-denominator diagnostic, got {err}"
    );
}

#[test]
fn mvdr_rejects_empty_covariance() {
    let cov = Array2::<Complex64>::zeros((0, 0));
    let steering = Array1::<Complex64>::zeros(0);

    let mvdr = MinimumVariance::default();
    let err = mvdr
        .compute_weights(&cov, &steering)
        .expect_err("should reject empty arrays");

    assert!(err.to_string().contains("non-empty"));
}

#[test]
fn mvdr_rejects_non_square_covariance() {
    let cov = Array2::<Complex64>::zeros((3, 4));
    let steering = Array1::<Complex64>::zeros(3);

    let mvdr = MinimumVariance::default();
    let err = mvdr
        .compute_weights(&cov, &steering)
        .expect_err("should reject non-square covariance");

    assert!(err.to_string().contains("square"));
}

#[test]
fn mvdr_rejects_dimension_mismatch() {
    let cov = test_utilities::create_test_covariance(4, 0.2, 0.1);
    let steering = Array1::zeros(5);

    let mvdr = MinimumVariance::default();
    let err = mvdr
        .compute_weights(&cov, &steering)
        .expect_err("should reject dimension mismatch");

    assert!(err.to_string().contains("dimension"));
}

#[test]
fn mvdr_rejects_negative_diagonal_loading() {
    let cov = test_utilities::create_test_covariance(4, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(4, 0.0);

    let mvdr = MinimumVariance::with_diagonal_loading(-0.1);
    let err = mvdr
        .compute_weights(&cov, &steering)
        .expect_err("should reject negative loading");

    assert!(err.to_string().contains("≥ 0") || err.to_string().contains(">= 0"));
}

#[test]
fn mvdr_rejects_nan_diagonal_loading() {
    let cov = test_utilities::create_test_covariance(4, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(4, 0.0);

    let mvdr = MinimumVariance::with_diagonal_loading(f64::NAN);
    let err = mvdr
        .compute_weights(&cov, &steering)
        .expect_err("should reject NaN loading");

    assert!(err.to_string().contains("finite"));
}

#[test]
fn mvdr_default_has_small_loading() {
    let mvdr = MinimumVariance::default();
    assert_eq!(mvdr.diagonal_loading, 1e-6);
}

#[test]
fn mvdr_new_has_zero_loading() {
    let mvdr = MinimumVariance::new();
    assert_eq!(mvdr.diagonal_loading, 0.0);
}
