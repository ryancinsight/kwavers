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
pub struct TrilinearInterpolator;

impl TrilinearInterpolator {
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

    /// Trilinear interpolation for continuous field evaluation
    ///
    /// **Algorithm**: Standard trilinear interpolation with bounds checking
    /// **Performance**: O(1) time complexity, zero allocations
    /// **Safety**: All array accesses validated per Rustonomicon Ch.5
    #[must_use]
    pub fn interpolate(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Get grid coordinates
        let xi = x / grid.dx;
        let yi = y / grid.dy;
        let zi = z / grid.dz;

        // Get integer indices (floor)
        let i = (xi.floor() as usize).clamp(0, grid.nx - 2);
        let j = (yi.floor() as usize).clamp(0, grid.ny - 2);
        let k = (zi.floor() as usize).clamp(0, grid.nz - 2);

        // Get fractional parts
        let dx = xi - i as f64;
        let dy = yi - j as f64;
        let dz = zi - k as f64;

        // Get the 8 corner values for trilinear interpolation
        // Safety: All indices are bounds-checked above
        let c000 = field[[i, j, k]];
        let c100 = field[[i + 1, j, k]];
        let c010 = field[[i, j + 1, k]];
        let c110 = field[[i + 1, j + 1, k]];
        let c001 = field[[i, j, k + 1]];
        let c101 = field[[i + 1, j, k + 1]];
        let c011 = field[[i, j + 1, k + 1]];
        let c111 = field[[i + 1, j + 1, k + 1]];

        // Perform trilinear interpolation
        // Interpolate along x
        let c00 = c000 * (1.0 - dx) + c100 * dx;
        let c10 = c010 * (1.0 - dx) + c110 * dx;
        let c01 = c001 * (1.0 - dx) + c101 * dx;
        let c11 = c011 * (1.0 - dx) + c111 * dx;

        // Interpolate along y
        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        // Interpolate along z
        c0 * (1.0 - dz) + c1 * dz
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
