//! Trilinear interpolation for heterogeneous media — delegates to `leto_ops::trilinear_index_space`.

use kwavers_grid::Grid;
use leto::Array3;
use leto_ops::trilinear_index_space;

/// Trilinear interpolator for heterogeneous media (thin wrapper over `leto_ops` SSOT).
#[derive(Debug)]
pub struct HetTrilinearInterpolator;

impl HetTrilinearInterpolator {
    /// Get grid indices for spatial coordinates with bounds checking.
    #[inline]
    pub fn get_indices(x: f64, y: f64, z: f64, grid: &Grid) -> (usize, usize, usize) {
        let ix = ((x / grid.dx).round() as usize).clamp(0, grid.nx - 1);
        let iy = ((y / grid.dy).round() as usize).clamp(0, grid.ny - 1);
        let iz = ((z / grid.dz).round() as usize).clamp(0, grid.nz - 1);
        (ix, iy, iz)
    }

    /// Trilinear interpolation for continuous field evaluation.
    ///
    /// Converts physical coordinates to fractional array indices and delegates to
    /// `leto_ops::trilinear_index_space` which handles degenerate axes (extent 1)
    /// and clamped out-of-bounds access.
    #[must_use]
    pub fn interpolate(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        trilinear_index_space(field.view(), x / grid.dx, y / grid.dy, z / grid.dz)
    }

    /// Get field value using appropriate interpolation method.
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

    /// Verify degenerate quasi-1D grid (ny=nz=1) doesn't panic and interpolates correctly.
    #[test]
    fn interpolate_quasi_1d_grid_does_not_panic_on_degenerate_y_z() {
        let grid = Grid::new(8, 1, 1, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let mut field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        for ix in 0..grid.nx {
            field[[ix, 0, 0]] = ix as f64;
        }
        for ix in 0..grid.nx {
            let v =
                HetTrilinearInterpolator::interpolate(&field, ix as f64 * grid.dx, 0.0, 0.0, &grid);
            assert!((v - ix as f64).abs() < 1e-12, "ix={ix} got {v}");
        }
        let v = HetTrilinearInterpolator::interpolate(&field, 0.5 * grid.dx, 0.0, 0.0, &grid);
        assert!((v - 0.5).abs() < 1e-12, "mid-cell: got {v}");
    }

    /// Quasi-2D: nz=1, non-degenerate nx and ny.
    #[test]
    fn interpolate_quasi_2d_grid_does_not_panic_on_degenerate_z() {
        let grid = Grid::new(4, 4, 1, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let mut field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        for ix in 0..grid.nx {
            for iy in 0..grid.ny {
                field[[ix, iy, 0]] = ix as f64 + 2.0 * iy as f64;
            }
        }
        let v =
            HetTrilinearInterpolator::interpolate(&field, 1.5 * grid.dx, 1.5 * grid.dy, 0.0, &grid);
        assert!((v - 4.5).abs() < 1e-12, "quasi-2D: got {v}");
    }
}
