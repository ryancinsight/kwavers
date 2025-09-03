//! Spatial phase randomization

use super::constants::{DEFAULT_SEED, MAX_PHASE_SHIFT};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Spatially-correlated phase randomization
#[derive(Debug)]
pub struct SpatialRandomization {
    correlation_length: f64,
    rng: ChaCha8Rng,
}

impl SpatialRandomization {
    /// Create new spatial randomizer
    #[must_use]
    pub fn new(correlation_length: f64) -> Self {
        Self {
            correlation_length,
            rng: ChaCha8Rng::seed_from_u64(DEFAULT_SEED),
        }
    }

    /// Generate spatially correlated random phases
    pub fn generate_correlated_phases(&mut self, positions: &Array2<f64>) -> Array1<f64> {
        let n_elements = positions.nrows();
        let mut phases = Array1::zeros(n_elements);

        // Generate uncorrelated random phases
        for i in 0..n_elements {
            phases[i] = self.rng.gen::<f64>() * MAX_PHASE_SHIFT;
        }

        // Apply spatial correlation using Gaussian kernel
        if self.correlation_length > 0.0 {
            self.apply_spatial_correlation(&mut phases, positions);
        }

        phases
    }

    /// Apply spatial correlation using Gaussian smoothing
    fn apply_spatial_correlation(&self, phases: &mut Array1<f64>, positions: &Array2<f64>) {
        let n = phases.len();
        let mut smoothed = Array1::zeros(n);
        let sigma2 = self.correlation_length * self.correlation_length;

        for i in 0..n {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..n {
                // Calculate distance between elements
                let dx = positions[[i, 0]] - positions[[j, 0]];
                let dy = positions[[i, 1]] - positions[[j, 1]];
                let dz = positions[[i, 2]] - positions[[j, 2]];
                let dist2 = dx * dx + dy * dy + dz * dz;

                // Gaussian weight
                let weight = (-dist2 / (2.0 * sigma2)).exp();
                weighted_sum += phases[j] * weight;
                weight_sum += weight;
            }

            smoothed[i] = weighted_sum / weight_sum;
        }

        phases.assign(&smoothed);
    }

    /// Set correlation length
    pub fn set_correlation_length(&mut self, length: f64) {
        self.correlation_length = length;
    }
}
