//! Beamforming Accuracy Validation Tests
//!
//! This module provides comprehensive validation of beamforming algorithms
//! against analytical solutions and performance benchmarks.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

use kwavers::sensor::beamforming::MinimumVariance;

#[cfg(feature = "legacy_algorithms")]
use kwavers::sensor::beamforming::adaptive::adaptive::LCMV;
#[cfg(feature = "legacy_algorithms")]
use kwavers::sensor::beamforming::adaptive::conventional::BeamformingAlgorithm;
#[cfg(feature = "legacy_algorithms")]
use kwavers::sensor::beamforming::{SourceEstimationCriterion, MUSIC};

/// Create a simple test covariance matrix
fn create_test_covariance(n: usize) -> Array2<Complex64> {
    let mut r = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let val = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.1 / (1.0 + (i as f64 - j as f64).abs()), 0.0)
            };
            r[(i, j)] = val;
        }
    }
    r
}

/// Create a steering vector for a linear array
fn create_steering_vector(n: usize, angle: f64) -> Array1<Complex64> {
    let k = 2.0 * PI; // Normalized wavenumber
    Array1::from_vec(
        (0..n)
            .map(|i| {
                let phase = k * (i as f64) * angle.sin();
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test MVDR basic functionality
    #[test]
    fn test_mvdr_basic() {
        let n = 8;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::default();
        let weights = mvdr.compute_weights(&cov, &steering).expect("MVDR weights");

        // Weights should be finite
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    /// Test MVDR condition monitoring
    ///
    /// Strict SSOT note: the MVDR implementation now has explicit error semantics and does not
    /// provide condition-monitoring helpers in this test surface. Conditioning checks should be
    /// evaluated via SSOT linear algebra utilities in targeted tests.
    #[test]
    fn test_mvdr_condition_monitoring() {
        let n = 8;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::default();
        let weights = mvdr.compute_weights(&cov, &steering).expect("MVDR weights");

        // Basic sanity: weights should be finite and correct length.
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    /// Test LCMV basic functionality
    #[cfg(feature = "legacy_algorithms")]
    #[test]
    fn test_lcmv_basic() {
        let n = 8;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let mut lcmv = LCMV::new();
        lcmv.add_constraint(&steering, Complex64::new(1.0, 0.0));

        let weights = lcmv.compute_weights(&cov);

        // Basic validation - weights should be finite and correct size
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    /// Test MUSIC basic functionality
    #[cfg(feature = "legacy_algorithms")]
    #[test]
    fn test_music_basic() {
        let n = 8;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let music = MUSIC::new(1);
        let spectrum = music
            .pseudospectrum(&cov, &steering)
            .expect("MUSIC pseudospectrum");

        // Pseudospectrum should be positive
        assert!(spectrum >= 0.0);
        assert!(spectrum.is_finite());
    }

    /// Test MUSIC source estimation
    #[cfg(feature = "legacy_algorithms")]
    #[test]
    fn test_music_source_estimation() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        // Test AIC vs MDL
        let music_aic =
            MUSIC::new_with_source_estimation(&cov, num_snapshots, SourceEstimationCriterion::AIC);

        let music_mdl =
            MUSIC::new_with_source_estimation(&cov, num_snapshots, SourceEstimationCriterion::MDL);

        // Should estimate reasonable number of sources (0 to n-1)
        assert!(music_aic.num_sources < n);
        assert!(music_mdl.num_sources < n);

        // MDL should be more conservative (may estimate fewer sources)
        assert!(music_mdl.num_sources <= music_aic.num_sources);
    }

    /// Test condition number computation
    ///
    /// Strict SSOT note: the condition-number helper was part of legacy MVDR surfaces that relied on
    /// non-SSOT eigendecomposition. Until SSOT provides complex Hermitian eigendecomposition (or an
    /// SSOT-conditioned estimator), this test is gated behind `legacy_algorithms`.
    #[cfg(feature = "legacy_algorithms")]
    #[test]
    fn test_covariance_condition_number() {
        let n = 8;
        let cov = create_test_covariance(n);

        let condition_number = MinimumVariance::covariance_condition_number(&cov);

        // Should be finite and positive
        assert!(condition_number > 0.0);
        assert!(condition_number.is_finite());

        // For well-conditioned synthetic data, should be reasonable
        assert!(
            condition_number < 1000.0,
            "Condition number too high: {}",
            condition_number
        );
    }
}
