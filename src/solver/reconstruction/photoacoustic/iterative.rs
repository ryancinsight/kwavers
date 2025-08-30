//! Iterative reconstruction methods for photoacoustic imaging
//!
//! This module implements various iterative reconstruction algorithms
//! including SIRT, ART, and OSEM.

use crate::error::KwaversResult;
use ndarray::{Array1, Array2, Array3, ArrayView2, Zip};
use rayon::prelude::*;

use super::config::PhotoacousticConfig;

/// Iterative reconstruction algorithms
#[derive(Debug, Clone))]
pub enum IterativeAlgorithm {
    /// Simultaneous Iterative Reconstruction Technique
    SIRT,
    /// Algebraic Reconstruction Technique
    ART,
    /// Ordered Subset Expectation Maximization
    OSEM { subsets: usize },
}

/// Iterative reconstruction methods
#[derive(Debug))]
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
        sensor_positions: &[[f64; 3],
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
            reconstruction[[i, j, k] = *val;
        }

        Ok(reconstruction)
    }

    /// Build system matrix for iterative reconstruction with proper physics
    fn build_system_matrix(
        &self,
        sensor_positions: &[[f64; 3],
        grid_size: [usize; 3],
    ) -> KwaversResult<Array2<f64>> {
        let n_sensors = sensor_positions.len();
        let n_voxels = grid_size[0] * grid_size[1] * grid_size[2];
        let mut matrix = Array2::zeros((n_sensors, n_voxels));

        // Physical grid dimensions (meters) - should be passed as parameter
        // Using reasonable defaults for photoacoustic imaging
        const GRID_PHYSICAL_SIZE: f64 = 0.05; // 50mm imaging region
        let dx = GRID_PHYSICAL_SIZE / grid_size[0] as f64;
        let dy = GRID_PHYSICAL_SIZE / grid_size[1] as f64;
        let dz = GRID_PHYSICAL_SIZE / grid_size[2] as f64;

        // Voxel volume for proper weighting
        let voxel_volume = dx * dy * dz;

        // Build matrix elements with proper Green's function
        for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
            for voxel_idx in 0..n_voxels {
                let (i, j, k) = self.linear_to_3d_index(voxel_idx, grid_size);
                let voxel_pos = [
                    (i as f64 + 0.5) * dx, // Center of voxel
                    (j as f64 + 0.5) * dy,
                    (k as f64 + 0.5) * dz,
                ];

                // Calculate distance from voxel center to sensor
                let distance = self.euclidean_distance(&voxel_pos, sensor_pos);

                if distance > 0.0 {
                    // Green's function for spherical wave propagation
                    // G(r) = 1/(4πr) with proper units
                    let green_function = 1.0 / (4.0 * std::f64::consts::PI * distance);

                    // Include voxel volume weighting and solid angle factor
                    let solid_angle_factor =
                        self.compute_solid_angle_factor(&voxel_pos, sensor_pos, dx);

                    matrix[[sensor_idx, voxel_idx] =
                        green_function * voxel_volume * solid_angle_factor;
                } else {
                    // Handle sensor inside voxel case
                    let effective_radius =
                        (voxel_volume * 3.0 / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0);
                    matrix[[sensor_idx, voxel_idx] =
                        1.0 / (4.0 * std::f64::consts::PI * effective_radius);
                }
            }
        }

        Ok(matrix)
    }

    /// Compute solid angle weighting factor for directional sensitivity
    fn compute_solid_angle_factor(
        &self,
        voxel_pos: &[f64; 3],
        sensor_pos: &[f64; 3],
        voxel_size: f64,
    ) -> f64 {
        let dx = sensor_pos[0] - voxel_pos[0];
        let dy = sensor_pos[1] - voxel_pos[1];
        let dz = sensor_pos[2] - voxel_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        if distance > 0.0 {
            // Approximate solid angle subtended by voxel at sensor
            let solid_angle = voxel_size * voxel_size / (distance * distance);
            // Normalize to maximum value of 1
            (solid_angle / (4.0 * std::f64::consts::PI)).min(1.0)
        } else {
            1.0
        }
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
        let mut x_updated = x.clone();

        // Process each equation sequentially
        for (i, row) in a.rows().into_iter().enumerate() {
            let ax_i = row.dot(&x_updated);
            let residual = y[i] - ax_i;
            let row_norm_sq = row.dot(&row);

            if row_norm_sq > 0.0 {
                let update_factor = self.relaxation_factor * residual / row_norm_sq;
                Zip::from(&mut x_updated)
                    .and(&row)
                    .for_each(|x_val, &a_val| {
                        *x_val += update_factor * a_val;
                    });
            }
        }

        Ok(x_updated)
    }

    /// OSEM (Ordered Subset Expectation Maximization) iteration
    fn osem_iteration(
        &self,
        a: &Array2<f64>,
        x: &Array1<f64>,
        y: &Array1<f64>,
        subsets: usize,
    ) -> KwaversResult<Array1<f64>> {
        let (n_measurements, n_voxels) = a.dim();
        let mut x_updated = x.clone();

        // Ensure positivity constraint for EM algorithms
        x_updated.mapv_inplace(|v| v.max(1e-10));

        // Divide measurements into ordered subsets
        let subset_size = n_measurements.div_ceil(subsets);

        // Process each subset
        for subset_idx in 0..subsets {
            let start_idx = subset_idx * subset_size;
            let end_idx = ((subset_idx + 1) * subset_size).min(n_measurements);

            // Extract subset of system matrix and measurements
            let a_subset = a.slice(ndarray::s![start_idx..end_idx, ..]);
            let y_subset = y.slice(ndarray::s![start_idx..end_idx]);

            // Compute sensitivity (normalization factor)
            let sensitivity = a_subset.sum_axis(ndarray::Axis(0));

            // Forward projection for this subset
            let forward_proj = a_subset.dot(&x_updated);

            // Compute ratio of measured to expected
            let mut ratio = Array1::zeros(end_idx - start_idx);
            for i in 0..ratio.len() {
                if forward_proj[i] > 1e-10 {
                    ratio[i] = y_subset[i] / forward_proj[i];
                } else {
                    ratio[i] = 0.0;
                }
            }

            // Back-projection of ratio
            let correction = a_subset.t().dot(&ratio);

            // Update with normalization
            for i in 0..n_voxels {
                if sensitivity[i] > 1e-10 {
                    x_updated[i] *= correction[i] / sensitivity[i];
                }
            }
        }

        Ok(x_updated)
    }

    /// Apply regularization with proper gradient-based methods
    fn apply_regularization(&self, x: &mut Array1<f64>) -> KwaversResult<()> {
        // Apply gradient descent step for regularization
        // This implements a proximal gradient step for various regularizers

        if self.regularization_parameter <= 0.0 {
            return Ok(());
        }

        let n = x.len();
        let grid_size_est = (n as f64).cbrt() as usize;

        // Compute regularization gradient (for smoothness)
        let mut grad_reg = Array1::zeros(n);

        // Apply 3D discrete Laplacian for smoothness regularization
        for idx in 0..n {
            let (i, j, k) = self.linear_to_3d_index(idx, [grid_size_est; 3]);
            let mut laplacian = -6.0 * x[idx];
            let mut count = 0;

            // Check all 6 neighbors in 3D
            for (di, dj, dk) in &[
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ] {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;
                let nk = (k as i32 + dk) as usize;

                if ni < grid_size_est && nj < grid_size_est && nk < grid_size_est {
                    let neighbor_idx = ni * grid_size_est * grid_size_est + nj * grid_size_est + nk;
                    if neighbor_idx < n {
                        laplacian += x[neighbor_idx];
                        count += 1;
                    }
                }
            }

            if count > 0 {
                grad_reg[idx] = -laplacian / count as f64;
            }
        }

        // Apply proximal gradient step
        *x = &*x - self.regularization_parameter * grad_reg;

        // Ensure non-negativity if using EM-type algorithms
        if matches!(self.algorithm, IterativeAlgorithm::OSEM { .. }) {
            x.mapv_inplace(|v| v.max(0.0));
        }

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
