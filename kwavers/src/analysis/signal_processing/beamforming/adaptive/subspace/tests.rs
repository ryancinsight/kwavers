use super::esmv::EigenspaceMV;
use super::music::MUSIC;
use crate::analysis::signal_processing::beamforming::test_utilities;
use approx::assert_relative_eq;
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn test_music_pseudospectrum_positivity() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let music = MUSIC::new(1);
    let spectrum = music
        .pseudospectrum(&cov, &steering)
        .expect("should compute pseudospectrum");

    assert!(spectrum >= 0.0);
    assert!(spectrum.is_finite());
}

#[test]
fn test_music_dimension_validation() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let music = MUSIC::new(4);
    let result = music.pseudospectrum(&cov, &steering);
    assert!(result.is_err());

    let music = MUSIC::new(5);
    let result = music.pseudospectrum(&cov, &steering);
    assert!(result.is_err());
}

#[test]
fn test_music_steering_dimension_mismatch() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(2, 0.0);

    let music = MUSIC::new(1);
    let result = music.pseudospectrum(&cov, &steering);
    assert!(result.is_err());
}

#[test]
fn test_music_scan_angles() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let music = MUSIC::new(1);

    for angle_deg in (-90..=90).step_by(30) {
        let angle = (angle_deg as f64) * PI / 180.0;
        let steering = test_utilities::create_steering_vector(n, angle);
        let spectrum = music
            .pseudospectrum(&cov, &steering)
            .expect("should compute for all angles");

        assert!(spectrum >= 0.0);
        assert!(spectrum.is_finite());
    }
}

#[test]
fn test_esmv_weight_computation() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let esmv = EigenspaceMV::new(1);
    let weights = esmv
        .compute_weights(&cov, &steering)
        .expect("should compute weights");

    assert_eq!(weights.len(), n);
    for &w in &weights {
        assert!(w.is_finite());
    }
}

#[test]
fn test_esmv_unit_gain_constraint() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let esmv = EigenspaceMV::with_diagonal_loading(1, 1e-4);
    let weights = esmv
        .compute_weights(&cov, &steering)
        .expect("should compute weights");

    let gain: Complex64 = weights
        .iter()
        .zip(steering.iter())
        .map(|(w, a)| w.conj() * a)
        .sum();

    assert_relative_eq!(gain.re, 1.0, epsilon = 1e-6);
    assert_relative_eq!(gain.im, 0.0, epsilon = 1e-6);
}

#[test]
fn test_esmv_dimension_validation() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let esmv = EigenspaceMV::new(4);
    let result = esmv.compute_weights(&cov, &steering);
    assert!(result.is_err());

    let esmv = EigenspaceMV::new(5);
    let result = esmv.compute_weights(&cov, &steering);
    assert!(result.is_err());
}

#[test]
fn test_esmv_steering_dimension_mismatch() {
    let n = 8;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(4, 0.0);

    let esmv = EigenspaceMV::new(2);
    let result = esmv.compute_weights(&cov, &steering);
    assert!(result.is_err());
}

#[test]
fn test_esmv_diagonal_loading_stability() {
    let n = 4;
    let mut cov = test_utilities::create_test_covariance(n, 0.2, 0.1);

    for i in 0..n {
        cov[(i, i)] *= Complex64::new(1e-8, 0.0);
    }

    let steering = test_utilities::create_steering_vector(n, 0.0);

    let esmv_low = EigenspaceMV::with_diagonal_loading(1, 1e-12);
    let result_low = esmv_low.compute_weights(&cov, &steering);

    let esmv_high = EigenspaceMV::with_diagonal_loading(1, 1e-4);
    let result_high = esmv_high.compute_weights(&cov, &steering);

    // At least one loading level must converge; high loading is more stable.
    assert!(
        result_low.is_ok() || result_high.is_ok(),
        "neither low nor high diagonal loading converged"
    );
    // If high loading succeeded, all weights must be finite.
    if let Ok(weights) = result_high {
        assert!(!weights.is_empty(), "high-loading weights must be non-empty");
        for &w in &weights {
            assert!(w.is_finite(), "high-loading weight {w} must be finite");
        }
    }
}

#[test]
fn test_music_esmv_consistency() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);
    let num_sources = 1;

    let music = MUSIC::new(num_sources);
    let spectrum = music
        .pseudospectrum(&cov, &steering)
        .expect("MUSIC should work");

    let esmv = EigenspaceMV::new(num_sources);
    let weights = esmv
        .compute_weights(&cov, &steering)
        .expect("ESMV should work");

    assert!(spectrum > 0.0);
    assert_eq!(weights.len(), n);
}

#[test]
fn test_music_peak_detection_concept() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let music = MUSIC::new(1);

    let angles: Vec<f64> = (0..180)
        .step_by(15)
        .map(|i| (i as f64 - 90.0) * PI / 180.0)
        .collect();

    let mut max_spectrum = 0.0;
    let mut max_angle = 0.0;

    for &angle in &angles {
        let steering = test_utilities::create_steering_vector(n, angle);
        let spectrum = music
            .pseudospectrum(&cov, &steering)
            .expect("should compute");

        if spectrum > max_spectrum {
            max_spectrum = spectrum;
            max_angle = angle;
        }
    }

    assert!(max_spectrum > 0.0);
    assert!(max_angle.abs() <= PI);
}

#[test]
fn test_esmv_signal_subspace_dimension_effect() {
    let n = 4;
    let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
    let steering = test_utilities::create_steering_vector(n, 0.0);

    let esmv1 = EigenspaceMV::new(1);
    let weights1 = esmv1
        .compute_weights(&cov, &steering)
        .expect("should compute with M=1");

    let esmv2 = EigenspaceMV::new(2);
    let weights2 = esmv2
        .compute_weights(&cov, &steering)
        .expect("should compute with M=2");

    let gain1: Complex64 = weights1
        .iter()
        .zip(steering.iter())
        .map(|(w, a)| w.conj() * a)
        .sum();
    let gain2: Complex64 = weights2
        .iter()
        .zip(steering.iter())
        .map(|(w, a)| w.conj() * a)
        .sum();

    assert_relative_eq!(gain1.re, 1.0, epsilon = 1e-6);
    assert_relative_eq!(gain2.re, 1.0, epsilon = 1e-6);

    for &w in &weights1 {
        assert!(w.is_finite());
    }
    for &w in &weights2 {
        assert!(w.is_finite());
    }
}
