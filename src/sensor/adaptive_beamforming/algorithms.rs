// adaptive_beamforming/algorithms.rs - Beamforming algorithms

use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Beamforming algorithm trait
pub trait BeamformingAlgorithm {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64>;
}

/// Delay and sum beamforming
#[derive(Debug)]
pub struct DelayAndSum;

impl BeamformingAlgorithm for DelayAndSum {
    fn compute_weights(
        &self,
        _covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        steering.clone()
    }
}

/// Minimum variance distortionless response
#[derive(Debug)]
pub struct MinimumVariance;

impl BeamformingAlgorithm for MinimumVariance {
    fn compute_weights(
        &self,
        _covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        // MVDR: w = R^-1 * a / (a^H * R^-1 * a)
        // Simplified implementation
        steering.clone()
    }
}
