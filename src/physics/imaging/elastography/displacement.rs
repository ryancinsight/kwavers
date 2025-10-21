//! Displacement Field Estimation for Shear Wave Tracking
//!
//! Tracks shear wave propagation by estimating tissue displacement from
//! ultrasound echo data.
//!
//! ## Methods
//!
//! - Cross-correlation: Standard speckle tracking
//! - Phase-based: Higher precision for small displacements
//! - Optical flow: Dense displacement fields
//!
//! ## References
//!
//! - Pinton, G. F., et al. (2006). "Rapid tracking of small displacements with
//!   ultrasound." *IEEE TUFFC*, 53(6), 1103-1117.
//! - Kasai, C., et al. (1985). "Real-time two-dimensional blood flow imaging
//!   using an autocorrelation technique." *IEEE Trans. Sonics Ultrason.*, 32(3), 458-464.

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;

/// Displacement field for shear wave tracking
#[derive(Debug, Clone)]
pub struct DisplacementField {
    /// Displacement in x direction (m)
    pub ux: Array3<f64>,
    /// Displacement in y direction (m)
    pub uy: Array3<f64>,
    /// Displacement in z direction (m)
    pub uz: Array3<f64>,
}

impl DisplacementField {
    /// Create zero displacement field
    pub fn zeros(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            ux: Array3::zeros((nx, ny, nz)),
            uy: Array3::zeros((nx, ny, nz)),
            uz: Array3::zeros((nx, ny, nz)),
        }
    }

    /// Calculate displacement magnitude at each point
    #[must_use]
    pub fn magnitude(&self) -> Array3<f64> {
        let (nx, ny, nz) = self.ux.dim();
        let mut mag = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let ux = self.ux[[i, j, k]];
                    let uy = self.uy[[i, j, k]];
                    let uz = self.uz[[i, j, k]];
                    mag[[i, j, k]] = (ux * ux + uy * uy + uz * uz).sqrt();
                }
            }
        }

        mag
    }
}

/// Displacement estimator for shear wave tracking
#[derive(Debug)]
pub struct DisplacementEstimator {
    /// Computational grid
    #[allow(dead_code)] // Used in future implementation
    grid: Grid,
    /// Estimation kernel size (points)
    #[allow(dead_code)] // Used in future implementation
    kernel_size: usize,
}

impl DisplacementEstimator {
    /// Create new displacement estimator
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    pub fn new(grid: &Grid) -> Self {
        Self {
            grid: grid.clone(),
            kernel_size: 5, // 5-point kernel for spatial derivatives
        }
    }

    /// Estimate displacement field from initial displacement
    ///
    /// # Arguments
    ///
    /// * `initial_displacement` - Initial displacement from push pulse
    ///
    /// # Returns
    ///
    /// Full displacement field with components
    ///
    /// # Note
    ///
    /// This simplified implementation assumes axial displacement dominates.
    /// Full implementation would use cross-correlation or phase tracking.
    pub fn estimate(&self, initial_displacement: &Array3<f64>) -> KwaversResult<DisplacementField> {
        let (nx, ny, nz) = initial_displacement.dim();

        // For now, assume dominant axial (z) displacement
        // Full implementation would use speckle tracking
        let uz = initial_displacement.clone();
        let ux = Array3::zeros((nx, ny, nz));
        let uy = Array3::zeros((nx, ny, nz));

        Ok(DisplacementField { ux, uy, uz })
    }

    /// Set kernel size for displacement estimation
    pub fn set_kernel_size(&mut self, size: usize) {
        self.kernel_size = size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_displacement_field_creation() {
        let field = DisplacementField::zeros(10, 10, 10);
        assert_eq!(field.ux.dim(), (10, 10, 10));
        assert_eq!(field.uy.dim(), (10, 10, 10));
        assert_eq!(field.uz.dim(), (10, 10, 10));
    }

    #[test]
    fn test_displacement_magnitude() {
        let mut field = DisplacementField::zeros(5, 5, 5);
        field.uz[[2, 2, 2]] = 1.0;

        let mag = field.magnitude();
        assert!((mag[[2, 2, 2]] - 1.0).abs() < 1e-10);
        assert!(mag[[0, 0, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_displacement_estimator() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let estimator = DisplacementEstimator::new(&grid);

        let initial = Array3::from_elem((20, 20, 20), 1.0e-6);
        let result = estimator.estimate(&initial);

        assert!(result.is_ok());
        let field = result.unwrap();
        assert_eq!(field.uz.dim(), (20, 20, 20));
    }
}
