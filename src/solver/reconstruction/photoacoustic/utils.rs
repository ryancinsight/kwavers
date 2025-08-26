//! Utility functions for photoacoustic reconstruction
//!
//! This module provides utility functions used across different
//! reconstruction algorithms.

use crate::error::KwaversResult;
use ndarray::{Array2, Array3, ArrayView2};

/// Utility functions for photoacoustic reconstruction
pub struct Utils;

impl Utils {
    /// Create new utils instance
    pub fn new() -> Self {
        Self
    }

    /// Calculate Euclidean distance between two 3D points
    pub fn euclidean_distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
    }

    /// Calculate back-projection weight for spherical spreading compensation
    pub fn calculate_back_projection_weight(
        &self,
        distance: f64,
        sound_speed: f64,
        dt: f64,
    ) -> f64 {
        // Weight includes:
        // - 1/r for spherical spreading
        // - Solid angle factor
        // - Time derivative factor
        if distance > 0.0 {
            let weight = 1.0 / (4.0 * std::f64::consts::PI * distance);
            weight * sound_speed * dt
        } else {
            0.0
        }
    }

    /// Propagate time-reversed signals (placeholder implementation)
    pub fn propagate_time_reversed_signals(
        &self,
        data: &mut Array3<f64>,
        _sound_speed: f64,
    ) -> KwaversResult<()> {
        // Time reversal implementation
        // This would involve solving the wave equation backward in time
        // For now, just reverse the time axis
        let (nx, ny, nt) = data.dim();
        for i in 0..nx {
            for j in 0..ny {
                for t in 0..nt / 2 {
                    let temp = data[[i, j, t]];
                    data[[i, j, t]] = data[[i, j, nt - 1 - t]];
                    data[[i, j, nt - 1 - t]] = temp;
                }
            }
        }
        Ok(())
    }

    /// Build forward model matrix for model-based reconstruction
    pub fn build_forward_model(
        &self,
        sensor_positions: &[[f64; 3]],
        grid_size: [usize; 3],
        sound_speed: f64,
        sampling_frequency: f64,
    ) -> KwaversResult<Array2<f64>> {
        let n_sensors = sensor_positions.len();
        let n_voxels = grid_size[0] * grid_size[1] * grid_size[2];
        let n_time_samples = (sampling_frequency * 0.001) as usize; // 1ms of data

        let mut forward_model = Array2::zeros((n_sensors * n_time_samples, n_voxels));

        // Grid spacing
        let dx = 1.0 / grid_size[0] as f64;
        let dy = 1.0 / grid_size[1] as f64;
        let dz = 1.0 / grid_size[2] as f64;
        let dt = 1.0 / sampling_frequency;

        // Build forward model
        for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
            for voxel_idx in 0..n_voxels {
                let k = voxel_idx % grid_size[2];
                let j = (voxel_idx / grid_size[2]) % grid_size[1];
                let i = voxel_idx / (grid_size[1] * grid_size[2]);

                let voxel_pos = [i as f64 * dx, j as f64 * dy, k as f64 * dz];
                let distance = self.euclidean_distance(&voxel_pos, sensor_pos);

                // Time of arrival
                let time_idx = (distance / sound_speed / dt) as usize;

                if time_idx < n_time_samples {
                    let row_idx = sensor_idx * n_time_samples + time_idx;
                    forward_model[[row_idx, voxel_idx]] =
                        1.0 / (4.0 * std::f64::consts::PI * distance.max(1e-10));
                }
            }
        }

        Ok(forward_model)
    }

    /// Solve regularized least squares problem
    pub fn solve_regularized_least_squares(
        &self,
        forward_model: &Array2<f64>,
        sensor_data: ArrayView2<f64>,
        regularization_parameter: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Simplified least squares solution
        // Actual implementation would use proper linear algebra solver

        let (n_measurements, n_voxels) = forward_model.dim();
        let grid_size = [(n_voxels as f64).cbrt() as usize; 3];

        // Flatten sensor data
        let y = sensor_data.as_slice().unwrap();

        // Solve (A^T A + λI)x = A^T y
        let at = forward_model.t();
        let ata = at.dot(forward_model);
        let aty = at.dot(&ndarray::Array1::from_vec(y.to_vec()));

        // Add regularization
        let mut ata_reg = ata;
        for i in 0..n_voxels {
            ata_reg[[i, i]] += regularization_parameter;
        }

        // Simple iterative solver (Jacobi method)
        let mut x = ndarray::Array1::zeros(n_voxels);
        for _iter in 0..50 {
            let mut x_new = x.clone();
            for i in 0..n_voxels {
                let mut sum = aty[i];
                for j in 0..n_voxels {
                    if i != j {
                        sum -= ata_reg[[i, j]] * x[j];
                    }
                }
                if ata_reg[[i, i]] != 0.0 {
                    x_new[i] = sum / ata_reg[[i, i]];
                }
            }
            x = x_new;
        }

        // Reshape to 3D
        let mut reconstruction = Array3::zeros((grid_size[0], grid_size[1], grid_size[2]));
        for (idx, val) in x.iter().enumerate() {
            let k = idx % grid_size[2];
            let j = (idx / grid_size[2]) % grid_size[1];
            let i = idx / (grid_size[1] * grid_size[2]);
            if i < grid_size[0] && j < grid_size[1] && k < grid_size[2] {
                reconstruction[[i, j, k]] = *val;
            }
        }

        Ok(reconstruction)
    }
}
