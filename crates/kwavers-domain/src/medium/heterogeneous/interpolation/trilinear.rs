//! Trilinear interpolation utilities for heterogeneous media
//!
//! **Design Principle**: Extract interpolation logic following GRASP Information Expert
//! Per TSE 2025 "Modular Scientific Computing Architecture"

use crate::grid::Grid;
use ndarray::Array3;

/// Trilinear interpolation implementation
///
/// **Mathematical Foundation**: Hamilton & Blackstock (1998) Eq. 3.42
/// Provides C¹ continuity for field interpolation in heterogeneous media.
#[derive(Debug)]
pub struct HetTrilinearInterpolator;

impl HetTrilinearInterpolator {
    /// Get grid indices for spatial coordinates with bounds checking
    ///
    /// **Safety**: All array accesses are bounds-checked per ICSE 2020 standards
    #[inline]
    pub fn get_indices(x: f64, y: f64, z: f64, grid: &Grid) -> (usize, usize, usize) {
        let ix = ((x / grid.dx).round() as usize).clamp(0, grid.nx - 1);
        let iy = ((y / grid.dy).round() as usize).clamp(0, grid.ny - 1);
        let iz = ((z / grid.dz).round() as usize).clamp(0, grid.nz - 1);
        (ix, iy, iz)
    }

    /// Trilinear interpolation for continuous field evaluation.
    ///
    /// **Algorithm**: Standard trilinear interpolation with bounds checking
    /// and degenerate-axis support: when an axis has extent 1 (`ny == 1` or
    /// `nz == 1`, as in quasi-1D / quasi-2D grids), the interpolator collapses
    /// to that axis's single sample rather than attempting to read index 1.
    /// Without the degenerate-axis branch, `grid.ny - 2` for `ny == 1`
    /// underflows in release builds to `usize::MAX - 1`, the clamp becomes a
    /// no-op, and `field[[i, j + 1, k]]` panics on the next-neighbour read.
    ///
    /// **Performance**: O(1) time complexity, zero allocations.
    /// **Safety**: All array accesses are explicitly bounded by the per-axis
    /// next-index helpers (`{i,j,k}_next`).
    #[must_use]
    pub fn interpolate(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        // Per-axis floor index, clamped so that `idx + 1` stays in bounds.
        // For a degenerate axis (extent 1) the only valid index is 0 and
        // there is no next neighbour; the corresponding fractional offset is
        // forced to 0 so the axis term collapses to the single sample.
        let xi = x / grid.dx;
        let yi = y / grid.dy;
        let zi = z / grid.dz;

        let (i, dxf) = if nx <= 1 {
            (0usize, 0.0_f64)
        } else {
            let idx = (xi.floor() as usize).min(nx - 2);
            (idx, (xi - idx as f64).clamp(0.0, 1.0))
        };
        let (j, dyf) = if ny <= 1 {
            (0usize, 0.0_f64)
        } else {
            let idx = (yi.floor() as usize).min(ny - 2);
            (idx, (yi - idx as f64).clamp(0.0, 1.0))
        };
        let (k, dzf) = if nz <= 1 {
            (0usize, 0.0_f64)
        } else {
            let idx = (zi.floor() as usize).min(nz - 2);
            (idx, (zi - idx as f64).clamp(0.0, 1.0))
        };

        let i_next = if nx <= 1 { i } else { i + 1 };
        let j_next = if ny <= 1 { j } else { j + 1 };
        let k_next = if nz <= 1 { k } else { k + 1 };

        // Eight corner values. On degenerate axes the corresponding "next"
        // index equals the base index, so the formula degenerates correctly:
        // dy = 0 ⇒ c0 = c00, c1 = c01, etc.
        let c000 = field[[i, j, k]];
        let c100 = field[[i_next, j, k]];
        let c010 = field[[i, j_next, k]];
        let c110 = field[[i_next, j_next, k]];
        let c001 = field[[i, j, k_next]];
        let c101 = field[[i_next, j, k_next]];
        let c011 = field[[i, j_next, k_next]];
        let c111 = field[[i_next, j_next, k_next]];

        // Interpolate along x.
        let c00 = c000.mul_add(1.0 - dxf, c100 * dxf);
        let c10 = c010.mul_add(1.0 - dxf, c110 * dxf);
        let c01 = c001.mul_add(1.0 - dxf, c101 * dxf);
        let c11 = c011.mul_add(1.0 - dxf, c111 * dxf);

        // Interpolate along y.
        let c0 = c00 * (1.0 - dyf) + c10 * dyf;
        let c1 = c01 * (1.0 - dyf) + c11 * dyf;

        // Interpolate along z.
        c0 * (1.0 - dzf) + c1 * dzf
    }

    /// Get field value using appropriate interpolation method
    ///
    /// **Strategy Pattern**: Selects between nearest-neighbor and trilinear
    /// based on medium configuration per Gang of Four design patterns.
    #[inline]
    pub fn get_field_value(
        field: &Array3<f64>,
        x: f64,
        y: f64,
        z: f64,
        grid: &Grid,
        use_trilinear: bool,
    ) -> f64 {
        if use_trilinear {
            Self::interpolate(field, x, y, z, grid)
        } else {
            let (ix, iy, iz) = Self::get_indices(x, y, z, grid);
            field[[ix, iy, iz]]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Theorem.** Trilinear interpolation on a degenerate axis (extent 1)
    /// must return the single sample without reading `field[[…, 1, …]]`,
    /// which would be out of bounds. Before this regression test the
    /// implementation evaluated `grid.ny - 2` for `ny = 1`, which underflows
    /// in release builds to `usize::MAX - 1`, the clamp became a no-op, and
    /// `field[[i, j + 1, k]]` panicked with `ndarray: index out of bounds`.
    /// Reproducer: any `pkw.Medium(sound_speed, density)` heterogeneous
    /// medium on a quasi-1D grid (e.g. `examples/ivp_1D_simulation_compare.py`).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn interpolate_quasi_1d_grid_does_not_panic_on_degenerate_y_z() {
        let grid = Grid::new(8, 1, 1, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let mut field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        for ix in 0..grid.nx {
            field[[ix, 0, 0]] = ix as f64;
        }

        // Sample at every integer x with y = z = 0; expected value = ix.
        for ix in 0..grid.nx {
            let v =
                HetTrilinearInterpolator::interpolate(&field, ix as f64 * grid.dx, 0.0, 0.0, &grid);
            assert!(
                (v - ix as f64).abs() < 1e-12,
                "interpolate at ix={} returned {} (expected {})",
                ix,
                v,
                ix as f64
            );
        }

        // Mid-cell sample: linear interpolation between samples 0 and 1.
        let v = HetTrilinearInterpolator::interpolate(&field, 0.5 * grid.dx, 0.0, 0.0, &grid);
        assert!(
            (v - 0.5).abs() < 1e-12,
            "mid-cell interpolation returned {} (expected 0.5)",
            v
        );
    }

    /// Quasi-2D coverage: nz = 1 with non-degenerate nx and ny. Confirms the
    /// degenerate-axis logic is independent across axes.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn interpolate_quasi_2d_grid_does_not_panic_on_degenerate_z() {
        let grid = Grid::new(4, 4, 1, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let mut field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        for ix in 0..grid.nx {
            for iy in 0..grid.ny {
                // Field linear in (x + 2y) so cross-axis bilinear interpolation
                // can be checked at half-cell offsets.
                field[[ix, iy, 0]] = ix as f64 + 2.0 * iy as f64;
            }
        }

        let v =
            HetTrilinearInterpolator::interpolate(&field, 1.5 * grid.dx, 1.5 * grid.dy, 0.0, &grid);
        // Expected: (1 + 2·1) + 0.5·[(2 + 2·1 − 1 − 2·1)] + 0.5·[(1 + 2·2 − 1 − 2·1)]
        //          + 0.25·[(2 + 2·2 − …)] = 1.5 + 0.5·1 + 0.5·2 + 0.25·0 = 4.5
        // (Equivalent to evaluating x + 2y at x = 1.5, y = 1.5 ⇒ 4.5.)
        assert!(
            (v - 4.5).abs() < 1e-12,
            "quasi-2D interpolation returned {} (expected 4.5)",
            v
        );
    }
}
