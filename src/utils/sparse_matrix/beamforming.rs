//! Beamforming-specific sparse matrix operations
//!
//! References:
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach"
//! - Li et al. (2003): "Robust Capon beamforming"

use super::csr::CompressedSparseRowMatrix;
use crate::error::KwaversResult;
use ndarray::{Array1, Array2};

/// Beamforming matrix operations
#[derive(Debug)]
pub struct BeamformingMatrix {
    /// Steering matrix (sparse)
    steering_matrix: CompressedSparseRowMatrix,
    /// Number of elements
    num_elements: usize,
    /// Number of directions
    num_directions: usize,
}

impl BeamformingMatrix {
    /// Create beamforming matrix
    #[must_use]
    pub fn create(num_elements: usize, num_directions: usize) -> Self {
        let steering_matrix = CompressedSparseRowMatrix::create(num_elements, num_directions);

        Self {
            steering_matrix,
            num_elements,
            num_directions,
        }
    }

    /// Build delay-and-sum steering matrix
    pub fn build_delay_sum_matrix(
        &mut self,
        element_positions: &Array2<f64>,
        directions: &Array2<f64>,
        frequency: f64,
        sound_speed: f64,
    ) -> KwaversResult<()> {
        use std::f64::consts::PI;

        let wavelength = sound_speed / frequency;
        let k = 2.0 * PI / wavelength;

        // Build sparse steering matrix
        let mut triplets = Vec::new();

        for (elem_idx, elem_pos) in element_positions.outer_iter().enumerate() {
            for (dir_idx, direction) in directions.outer_iter().enumerate() {
                // Compute phase delay
                let delay = elem_pos.dot(&direction) / sound_speed;
                let phase = k * sound_speed * delay;

                // Complex exponential weight (store real part for now)
                let weight = phase.cos();

                if weight.abs() > 1e-10 {
                    triplets.push((elem_idx, dir_idx, weight));
                }
            }
        }

        // Convert to CSR
        let mut coo = super::coo::CoordinateMatrix::create(self.num_elements, self.num_directions);
        for (row, col, val) in triplets {
            coo.add_triplet(row, col, val);
        }

        self.steering_matrix = coo.to_csr();
        Ok(())
    }

    /// Apply beamforming weights
    pub fn apply_weights(&self, data: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        self.steering_matrix.multiply_vector(data.view())
    }

    /// Compute covariance matrix with proper sample covariance calculation
    /// Reference: Van Trees (2002) "Optimum Array Processing", Eq. 6.26  
    #[must_use]
    pub fn compute_covariance(
        &self,
        data: &Array2<f64>,
        diagonal_loading: f64,
    ) -> CompressedSparseRowMatrix {
        let (n_elements, n_snapshots) = data.dim();

        // Use coordinate matrix for efficient construction
        let mut coo = super::coo::CoordinateMatrix::create(n_elements, n_elements);

        // Compute sample covariance: R = (1/N) * X * X^T + λI
        for i in 0..n_elements {
            for j in i..n_elements {
                // Only upper triangular due to symmetry
                let mut sum = 0.0;

                // Compute cross-correlation between elements i and j
                for k in 0..n_snapshots {
                    sum += data[[i, k]] * data[[j, k]];
                }

                let value = sum / n_snapshots as f64;

                // Add diagonal loading to diagonal elements
                let final_value = if i == j {
                    value + diagonal_loading
                } else {
                    value
                };

                if final_value.abs() > 1e-14 {
                    coo.add_triplet(i, j, final_value);

                    // Add symmetric element for off-diagonal
                    if i != j {
                        coo.add_triplet(j, i, final_value);
                    }
                }
            }
        }

        coo.to_csr()
    }

    /// Get steering matrix
    #[must_use]
    pub fn steering_matrix(&self) -> &CompressedSparseRowMatrix {
        &self.steering_matrix
    }
}
