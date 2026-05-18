//! Iterative reconstruction methods for photoacoustic imaging.
//!
//! Implements SIRT, ART, and OSEM algorithms.
//! References: Kak & Slaney (1988) "Principles of Computerized Tomographic Imaging".

mod iterations;
mod system;

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array3, ArrayView2};

use super::algorithms::PhotoacousticAlgorithm;
use super::config::ReconstructionPhotoacousticConfig;

/// Iterative reconstruction algorithm selector.
#[derive(Debug, Clone)]
pub enum IterativeAlgorithm {
    /// Simultaneous Iterative Reconstruction Technique.
    SIRT,
    /// Algebraic Reconstruction Technique.
    ART,
    /// Ordered Subset Expectation Maximization.
    OSEM { subsets: usize },
}

/// Iterative reconstruction methods handler.
#[derive(Debug)]
pub struct IterativeMethods {
    pub(super) algorithm: IterativeAlgorithm,
    pub(super) iterations: usize,
    pub(super) relaxation_factor: f64,
    pub(super) regularization_parameter: f64,
}

impl IterativeMethods {
    /// Create new iterative methods handler from photoacoustic config.
    pub fn new(config: &ReconstructionPhotoacousticConfig) -> Self {
        let (algorithm, iterations, relaxation_factor) =
            if let PhotoacousticAlgorithm::Iterative {
                algorithm,
                iterations,
                relaxation_factor,
            } = &config.algorithm
            {
                (algorithm.clone(), *iterations, *relaxation_factor)
            } else {
                (IterativeAlgorithm::SIRT, 50, 0.5)
            };

        Self {
            algorithm,
            iterations,
            relaxation_factor,
            regularization_parameter: config.regularization_parameter,
        }
    }

    /// Perform iterative reconstruction.
    ///
    /// Builds the system matrix, then runs the selected algorithm for
    /// `self.iterations` steps, applying regularization after each step.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn reconstruct(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
        grid_size: [usize; 3],
    ) -> KwaversResult<Array3<f64>> {
        let system_matrix = self.build_system_matrix(sensor_positions, grid_size)?;

        let mut reconstruction = Array3::zeros((grid_size[0], grid_size[1], grid_size[2]));
        let n_voxels = grid_size[0] * grid_size[1] * grid_size[2];
        let mut x = Array1::zeros(n_voxels);

        let y = sensor_data.as_slice().unwrap().to_vec();
        let y = Array1::from_vec(y);

        for _iter in 0..self.iterations {
            match &self.algorithm {
                IterativeAlgorithm::SIRT => {
                    x = self.sirt_iteration(&system_matrix, &x, &y)?;
                }
                IterativeAlgorithm::ART => {
                    self.art_iteration(&system_matrix, &mut x, &y)?;
                }
                IterativeAlgorithm::OSEM { subsets } => {
                    self.osem_iteration(&system_matrix, &mut x, &y, *subsets)?;
                }
            }

            if self.regularization_parameter > 0.0 {
                self.apply_regularization(&mut x)?;
            }
        }

        for (idx, val) in x.iter().enumerate() {
            let (i, j, k) = self.linear_to_3d_index(idx, grid_size);
            reconstruction[[i, j, k]] = *val;
        }

        Ok(reconstruction)
    }
}
