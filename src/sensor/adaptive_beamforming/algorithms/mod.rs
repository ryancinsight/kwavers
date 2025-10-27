//! Advanced Beamforming Algorithms
//!
//! This module implements state-of-the-art beamforming algorithms including:
//! - Delay-and-Sum (conventional beamforming)
//! - MVDR (Capon beamformer - Minimum Variance Distortionless Response)
//! - MUSIC (Multiple Signal Classification)
//! - Eigenspace-based Minimum Variance (ESPMV)
//!
//! # References
//! - Capon (1969), "High-resolution frequency-wavenumber spectrum analysis"
//! - Schmidt (1986), "Multiple emitter location and signal parameter estimation"
//! - Van Trees (2002), "Optimum Array Processing"
//! - Stoica & Nehorai (1990), "MUSIC, maximum likelihood, and Cramer-Rao bound"

use ndarray::{Array1, Array2};
use num_complex::Complex64;

// Internal utility functions
mod utils;

// Algorithm implementations
mod covariance_taper;
mod delay_and_sum;
mod eigenspace_mv;
mod music;
mod mvdr;
mod robust_capon;
mod source_estimation;

// Re-export main types
pub use covariance_taper::{CovarianceTaper, TaperType};
pub use delay_and_sum::DelayAndSum;
pub use eigenspace_mv::EigenspaceMV;
pub use music::MUSIC;
pub use mvdr::MinimumVariance;
pub use robust_capon::RobustCapon;
pub use source_estimation::{estimate_num_sources, SourceEstimationCriterion};

/// Beamforming algorithm trait
pub trait BeamformingAlgorithm {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64>;
}

// Re-export subspace trackers and related types from the old module
// These are temporarily exported from the old file until they are refactored
pub use super::algorithms_old::{OrthonormalSubspaceTracker, SubspaceTracker};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

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

    #[test]
    fn test_delay_and_sum() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = DelayAndSum;
        let weights = beamformer.compute_weights(&cov, &steering);

        // Weights should equal steering vector
        for i in 0..n {
            assert_relative_eq!(weights[i].re, steering[i].re, epsilon = 1e-10);
            assert_relative_eq!(weights[i].im, steering[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mvdr_weights_exist() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = MinimumVariance::default();
        let weights = beamformer.compute_weights(&cov, &steering);

        // Weights should be finite
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_mvdr_unit_gain_constraint() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = MinimumVariance::default();
        let weights = beamformer.compute_weights(&cov, &steering);

        // Check unit gain constraint: w^H a = 1
        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(w, a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.norm(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_music_pseudospectrum() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let music = MUSIC::new(1); // 1 source
        let spectrum = music.pseudospectrum(&cov, &steering);

        // Pseudospectrum should be positive
        assert!(spectrum >= 0.0);
        assert!(spectrum.is_finite());
    }
}
