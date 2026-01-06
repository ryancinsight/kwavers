//! Delay-and-Sum (conventional) beamforming algorithm

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::BeamformingAlgorithm;

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
        _covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> crate::error::KwaversResult<Array1<Complex64>> {
        // Conventional beamforming: w = a (steering vector)
        Ok(steering.clone())
    }
}
