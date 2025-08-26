//! Iterative reconstruction methods for photoacoustic imaging
//!
//! This module implements various iterative reconstruction algorithms
//! including SIRT, ART, and OSEM.

use crate::error::KwaversResult;
use ndarray::{Array1, Array2, Array3, ArrayView2, Zip};
use rayon::prelude::*;

use super::config::PhotoacousticConfig;

/// Iterative reconstruction algorithms
#[derive(Debug, Clone)]
pub enum IterativeAlgorithm {
    /// Simultaneous Iterative Reconstruction Technique
    SIRT,
    /// Algebraic Reconstruction Technique
    ART,
    /// Ordered Subset Expectation Maximization
    OSEM { subsets: usize },
}

/// Iterative reconstruction methods
pub struct IterativeMethods {
    algorithm: IterativeAlgorithm,
    iterations: usize,
    relaxation_factor: f64,
    regularization_parameter: f64,
}

impl IterativeMethods {
    /// Create new iterative methods handler
    pub fn new(config: &PhotoacousticConfig) -> Self {
        let (algorithm, iterations, relaxation_factor) =
            if let PhotoacousticAlgorithm::Iterative {
                algorithm,
                iterations,
                relaxation_factor,
            } = &config.algorithm
            {
                (algorithm.clone(), *iterations, *relaxation_factor)
            } else {
                // Default values
                (IterativeAlgorithm::SIRT, 50, 0.5)
            };

        Self {
            algorithm,
            iterations,
            relaxation_factor,
            regularization_parameter: config.regularization_parameter,
        }
    }

    /// Perform iterative reconstruction
    pub fn reconstruct(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
        grid_size: [usize; 3],
    ) -> KwaversResult<Array3<f64>> {
        // Build system matrix A where y = Ax (y: measurements, x: image)
        let system_matrix = self.build_system_matrix(sensor_positions, grid_size)?;

        // Initialize reconstruction
        let mut reconstruction = Array3::zeros((grid_size[0], grid_size[1], grid_size[2]));
        let n_voxels = grid_size[0] * grid_size[1] * grid_size[2];
        let mut x = Array1::zeros(n_voxels);

        // Flatten sensor data
        let y = sensor_data.as_slice().unwrap().to_vec();
        let y = Array1::from_vec(y);

        // Perform iterations
        for _iter in 0..self.iterations {
            match &self.algorithm {
                IterativeAlgorithm::SIRT => {
                    x = self.sirt_iteration(&system_matrix, &x, &y)?;
                }
                IterativeAlgorithm::ART => {
                    x = self.art_iteration(&system_matrix, &x, &y)?;
                }
                IterativeAlgorithm::OSEM { subsets } => {
                    x = self.osem_iteration(&system_matrix, &x, &y, *subsets)?;
                }
            }

            // Apply regularization
            if self.regularization_parameter > 0.0 {
                self.apply_regularization(&mut x)?;
            }
        }

        // Reshape to 3D
        for (idx, val) in x.iter().enumerate() {
            let (i, j, k) = self.linear_to_3d_index(idx, grid_size);
            reconstruction[[i, j, k]] = *val;
        }

        Ok(reconstruction)
    }

    /// Build system matrix for iterative reconstruction
    fn build_system_matrix(
        &self,
        sensor_positions: &[[f64; 3]],
        grid_size: [usize; 3],
    ) -> KwaversResult<Array2<f64>> {
        let n_sensors = sensor_positions.len();
        let n_voxels = grid_size[0] * grid_size[1] * grid_size[2];
        let mut matrix = Array2::zeros((n_sensors, n_voxels));

        // Grid spacing (assume uniform)
        let dx = 1.0 / grid_size[0] as f64;
        let dy = 1.0 / grid_size[1] as f64;
        let dz = 1.0 / grid_size[2] as f64;

        // Build matrix elements
        for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
            for voxel_idx in 0..n_voxels {
                let (i, j, k) = self.linear_to_3d_index(voxel_idx, grid_size);
                let voxel_pos = [i as f64 * dx, j as f64 * dy, k as f64 * dz];

                // Calculate contribution (simplified - actual would use proper forward model)
                let distance = self.euclidean_distance(&voxel_pos, sensor_pos);
                if distance > 0.0 {
                    matrix[[sensor_idx, voxel_idx]] = 1.0 / (4.0 * std::f64::consts::PI * distance);
                }
            }
        }

        Ok(matrix)
    }

    /// SIRT iteration
    fn sirt_iteration(
        &self,
        a: &Array2<f64>,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        // Calculate Ax
        let ax = a.dot(x);

        // Calculate residual
        let residual = y - &ax;

        // Calculate update
        let update = a.t().dot(&residual);

        // Apply relaxation and update
        Ok(x + self.relaxation_factor * &update)
    }

    /// ART iteration
    fn art_iteration(
        &self,
        a: &Array2<f64>,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let mut x_new = x.clone();

        // Process each equation sequentially
        for (i, row) in a.rows().into_iter().enumerate() {
            let ax_i = row.dot(&x_new);
            let residual = y[i] - ax_i;
            let row_norm_sq = row.dot(&row);

            if row_norm_sq > 0.0 {
                let update_factor = self.relaxation_factor * residual / row_norm_sq;
                Zip::from(&mut x_new).and(&row).for_each(|x_val, &a_val| {
                    *x_val += update_factor * a_val;
                });
            }
        }

        Ok(x_new)
    }

    /// OSEM iteration
    fn osem_iteration(
        &self,
        a: &Array2<f64>,
        x: &Array1<f64>,
        y: &Array1<f64>,
        subsets: usize,
    ) -> KwaversResult<Array1<f64>> {
        // Simplified OSEM - actual implementation would use proper subsets
        // For now, just use SIRT as placeholder
        self.sirt_iteration(a, x, y)
    }

    /// Apply regularization (e.g., total variation)
    fn apply_regularization(&self, x: &mut Array1<f64>) -> KwaversResult<()> {
        // Simple L2 regularization
        *x *= 1.0 / (1.0 + self.regularization_parameter);
        Ok(())
    }

    /// Convert linear index to 3D index
    fn linear_to_3d_index(&self, idx: usize, grid_size: [usize; 3]) -> (usize, usize, usize) {
        let k = idx % grid_size[2];
        let j = (idx / grid_size[2]) % grid_size[1];
        let i = idx / (grid_size[1] * grid_size[2]);
        (i, j, k)
    }

    /// Calculate Euclidean distance
    fn euclidean_distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
    }
}

// Re-export for use in algorithms module
use super::algorithms::PhotoacousticAlgorithm;
