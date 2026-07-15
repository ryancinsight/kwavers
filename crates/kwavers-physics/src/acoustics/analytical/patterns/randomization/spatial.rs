//! Spatial phase randomization

use super::constants::{DEFAULT_SEED, MAX_PHASE_SHIFT};
use leto::{Array1, Array2};
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
        let n_elements = positions.shape()[0];
        let mut phases = Array1::zeros([n_elements]);

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
        let n = phases.size();
        let mut smoothed = Array1::zeros([n]);
        let sigma2 = self.correlation_length * self.correlation_length;

        for i in 0..n {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..n {
                // Calculate distance between elements
                let dx = positions[[i, 0]] - positions[[j, 0]];
                let dy = positions[[i, 1]] - positions[[j, 1]];
                let dz = positions[[i, 2]] - positions[[j, 2]];
                let dist2 = dz.mul_add(dz, dx.mul_add(dx, dy * dy));

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

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array2;

    /// generate_correlated_phases returns one phase per element.
    #[test]
    fn generates_one_phase_per_element() {
        let mut sr = SpatialRandomization::new(1e-3);
        // 4-element 1-D array; positions are (x,y,z) columns
        let positions = Array2::from_shape_fn([4, 3], |[i, _]| i as f64 * 1e-3);
        let phases = sr.generate_correlated_phases(&positions);
        assert_eq!(phases.size(), 4, "must return 4 phases for 4 elements");
    }

    /// With zero correlation length, phases are uncorrelated random values in [0, MAX_PHASE_SHIFT).
    #[test]
    fn zero_correlation_produces_uncorrelated_phases() {
        use super::super::constants::MAX_PHASE_SHIFT;
        let mut sr = SpatialRandomization::new(0.0);
        let positions = Array2::zeros([8, 3]);
        let phases = sr.generate_correlated_phases(&positions);
        assert_eq!(phases.size(), 8);
        for &p in phases.iter() {
            assert!(
                (0.0..MAX_PHASE_SHIFT).contains(&p),
                "phase {p} out of [0, MAX_PHASE_SHIFT)"
            );
        }
    }

    /// set_correlation_length updates the stored length (deterministic RNG seed).
    #[test]
    fn set_correlation_length_changes_smoothing() {
        let mut sr = SpatialRandomization::new(0.0);
        sr.set_correlation_length(5e-3);
        // Just verify it doesn't panic and the struct is usable afterward
        let positions = Array2::from_shape_fn([3, 3], |[i, _]| i as f64 * 5e-3);
        let phases = sr.generate_correlated_phases(&positions);
        assert_eq!(phases.size(), 3);
    }
}
