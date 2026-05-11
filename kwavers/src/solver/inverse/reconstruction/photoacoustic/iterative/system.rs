//! System matrix construction and geometric helpers for [`IterativeMethods`].

use super::IterativeMethods;
use crate::core::error::KwaversResult;
use ndarray::Array2;

impl IterativeMethods {
    /// Build system matrix A where y = Ax (y: measurements, x: image).
    ///
    /// Each entry A[s, v] = Green's function G(r_sv) × voxel_volume × solid_angle(r_sv),
    /// using G(r) = 1/(4πr) for spherical wave propagation.
    ///
    /// Physical grid is assumed to span GRID_PHYSICAL_SIZE = 50 mm.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn build_system_matrix(
        &self,
        sensor_positions: &[[f64; 3]],
        grid_size: [usize; 3],
    ) -> KwaversResult<Array2<f64>> {
        let n_sensors = sensor_positions.len();
        let n_voxels = grid_size[0] * grid_size[1] * grid_size[2];
        let mut matrix = Array2::zeros((n_sensors, n_voxels));

        const GRID_PHYSICAL_SIZE: f64 = 0.05; // 50 mm imaging region
        let dx = GRID_PHYSICAL_SIZE / grid_size[0] as f64;
        let dy = GRID_PHYSICAL_SIZE / grid_size[1] as f64;
        let dz = GRID_PHYSICAL_SIZE / grid_size[2] as f64;

        let voxel_volume = dx * dy * dz;

        for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
            for voxel_idx in 0..n_voxels {
                let (i, j, k) = self.linear_to_3d_index(voxel_idx, grid_size);
                let voxel_pos = [
                    (i as f64 + 0.5) * dx,
                    (j as f64 + 0.5) * dy,
                    (k as f64 + 0.5) * dz,
                ];

                let distance = self.euclidean_distance(&voxel_pos, sensor_pos);

                if distance > 0.0 {
                    let green_function = 1.0 / (4.0 * std::f64::consts::PI * distance);
                    let solid_angle_factor =
                        self.compute_solid_angle_factor(&voxel_pos, sensor_pos, dx);
                    matrix[[sensor_idx, voxel_idx]] =
                        green_function * voxel_volume * solid_angle_factor;
                } else {
                    let effective_radius =
                        (voxel_volume * 3.0 / (4.0 * std::f64::consts::PI)).cbrt();
                    matrix[[sensor_idx, voxel_idx]] =
                        1.0 / (4.0 * std::f64::consts::PI * effective_radius);
                }
            }
        }

        Ok(matrix)
    }

    /// Compute solid angle weighting factor for directional sensitivity.
    ///
    /// Returns Ω/4π clamped to [0, 1], where Ω ≈ voxel_size² / r².
    pub(super) fn compute_solid_angle_factor(
        &self,
        voxel_pos: &[f64; 3],
        sensor_pos: &[f64; 3],
        voxel_size: f64,
    ) -> f64 {
        let dx = sensor_pos[0] - voxel_pos[0];
        let dy = sensor_pos[1] - voxel_pos[1];
        let dz = sensor_pos[2] - voxel_pos[2];
        let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

        if distance > 0.0 {
            let solid_angle = voxel_size * voxel_size / (distance * distance);
            (solid_angle / (4.0 * std::f64::consts::PI)).min(1.0)
        } else {
            1.0
        }
    }

    /// Convert linear voxel index to (i, j, k) 3-D index.
    ///
    /// Row-major order: k varies fastest, i slowest.
    pub(super) fn linear_to_3d_index(
        &self,
        idx: usize,
        grid_size: [usize; 3],
    ) -> (usize, usize, usize) {
        let k = idx % grid_size[2];
        let j = (idx / grid_size[2]) % grid_size[1];
        let i = idx / (grid_size[1] * grid_size[2]);
        (i, j, k)
    }

    /// Euclidean distance between two 3-D points.
    pub(super) fn euclidean_distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        (p1[2] - p2[2])
            .mul_add(
                p1[2] - p2[2],
                (p1[1] - p2[1]).mul_add(p1[1] - p2[1], (p1[0] - p2[0]).powi(2)),
            )
            .sqrt()
    }
}
