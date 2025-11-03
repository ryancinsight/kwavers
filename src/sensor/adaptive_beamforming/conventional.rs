//! Conventional beamforming algorithms
//!
//! This module implements traditional beamforming methods that form the
//! foundation for more advanced adaptive techniques.

use ndarray::Array1;
use num_complex::Complex64;

/// Beamforming algorithm trait
pub trait BeamformingAlgorithm {
    fn compute_weights(
        &self,
        covariance: &ndarray::Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64>;
}

/// Delay and sum beamforming (conventional beamforming)
///
/// The simplest beamforming algorithm that applies uniform weighting
/// to all array elements after steering delays.
///
/// # References
/// - Van Veen & Buckley (1988), "Beamforming: A versatile approach to spatial filtering"
#[derive(Debug)]
pub struct DelayAndSum;

impl BeamformingAlgorithm for DelayAndSum {
    fn compute_weights(
        &self,
        _covariance: &ndarray::Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        // Conventional beamforming: w = a (steering vector)
        steering.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;
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
}
