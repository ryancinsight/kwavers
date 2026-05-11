//! Linear (trilinear) interpolation for inter-domain coupling.

use super::InterpolationManager;
use crate::core::error::KwaversResult;
use ndarray::Array3;

impl InterpolationManager {
    /// Trilinear interpolation from source to target grid.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn linear_interpolation(
        &self,
        source_field: &Array3<f64>,
        source_coords: &[(f64, f64, f64)],
        target_coords: &[(f64, f64, f64)],
    ) -> KwaversResult<Array3<f64>> {
        let shape = source_field.shape();
        let mut result = Array3::zeros([shape[0], shape[1], shape[2]]);

        if source_coords.is_empty() || target_coords.is_empty() {
            return Ok(result);
        }

        // Build bounding box for source grid
        let (min_x, max_x) = source_coords
            .iter()
            .fold((f64::MAX, f64::MIN), |(min, max), &(x, _, _)| {
                (min.min(x), max.max(x))
            });
        let (min_y, max_y) = source_coords
            .iter()
            .fold((f64::MAX, f64::MIN), |(min, max), &(_, y, _)| {
                (min.min(y), max.max(y))
            });
        let (min_z, max_z) = source_coords
            .iter()
            .fold((f64::MAX, f64::MIN), |(min, max), &(_, _, z)| {
                (min.min(z), max.max(z))
            });

        let dx = (max_x - min_x) / (shape[0] as f64 - 1.0);
        let dy = (max_y - min_y) / (shape[1] as f64 - 1.0);
        let dz = (max_z - min_z) / (shape[2] as f64 - 1.0);

        // For simplicity, assume uniform grid spacing and map target coords to source indices
        for (idx, &(tx, ty, tz)) in target_coords.iter().enumerate().take(result.len()) {
            // Find fractional indices in source grid
            let fi = ((tx - min_x) / dx).max(0.0).min((shape[0] - 2) as f64);
            let fj = ((ty - min_y) / dy).max(0.0).min((shape[1] - 2) as f64);
            let fk = ((tz - min_z) / dz).max(0.0).min((shape[2] - 2) as f64);

            let i0 = fi.floor() as usize;
            let j0 = fj.floor() as usize;
            let k0 = fk.floor() as usize;
            let i1 = (i0 + 1).min(shape[0] - 1);
            let j1 = (j0 + 1).min(shape[1] - 1);
            let k1 = (k0 + 1).min(shape[2] - 1);

            // Interpolation weights
            let wx = fi - (i0 as f64);
            let wy = fj - (j0 as f64);
            let wz = fk - (k0 as f64);

            // Trilinear interpolation: c000*(1-wx)*(1-wy)*(1-wz) + c100*wx*(1-wy)*(1-wz) + ...
            let c000 = source_field[[i0, j0, k0]];
            let c100 = source_field[[i1, j0, k0]];
            let c010 = source_field[[i0, j1, k0]];
            let c110 = source_field[[i1, j1, k0]];
            let c001 = source_field[[i0, j0, k1]];
            let c101 = source_field[[i1, j0, k1]];
            let c011 = source_field[[i0, j1, k1]];
            let c111 = source_field[[i1, j1, k1]];

            let value = (c111 * wx * wy).mul_add(wz, (c011 * (1.0 - wx) * wy).mul_add(wz, (c101 * wx * (1.0 - wy)).mul_add(wz, (c001 * (1.0 - wx) * (1.0 - wy)).mul_add(wz, (c110 * wx * wy).mul_add(1.0 - wz, (c010 * (1.0 - wx) * wy).mul_add(1.0 - wz, (c000 * (1.0 - wx) * (1.0 - wy)).mul_add(1.0 - wz, c100 * wx * (1.0 - wy) * (1.0 - wz))))))));

            // Map to result array (assuming same shape)
            let ri = idx / (shape[1] * shape[2]);
            let rj = (idx / shape[2]) % shape[1];
            let rk = idx % shape[2];
            if ri < shape[0] && rj < shape[1] && rk < shape[2] {
                result[[ri, rj, rk]] = value;
            }
        }

        Ok(result)
    }
}
