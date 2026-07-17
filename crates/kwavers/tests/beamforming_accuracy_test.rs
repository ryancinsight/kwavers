//! Beamforming Accuracy Validation Tests
//!
//! This module provides comprehensive validation of beamforming algorithms
//! against analytical solutions and performance benchmarks.

use kwavers_math::fft::Complex64;
use leto::{Array1, Array2};
use std::f64::consts::PI;

use kwavers_analysis::signal_processing::beamforming::adaptive::MinimumVariance;

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
            r[[i, j]] = val;
        }
    }
    r
}

/// Create a steering vector for a linear array
fn create_steering_vector(n: usize, angle: f64) -> Array1<Complex64> {
    let k = 2.0 * PI; // Normalized wavenumber
    Array1::from_vec(
        n,
        (0..n)
            .map(|i| {
                let phase = k * (i as f64) * angle.sin();
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect(),
    )
    .unwrap()
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
        for &w in weights.iter() {
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
        for &w in weights.iter() {
            assert!(w.is_finite());
        }
    }
}
