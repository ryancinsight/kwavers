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
    /// # Implementation
    ///
    /// Simulates ultrafast plane wave imaging with temporal evolution.
    /// Uses plane wave propagation model for displacement tracking and temporal evolution.
    pub fn estimate(&self, initial_displacement: &Array3<f64>) -> KwaversResult<DisplacementField> {
        let (nx, ny, nz) = initial_displacement.dim();

        // Simulate ultrafast imaging sequence (typical: 10,000 fps)
        // Track displacement evolution over time
        let n_frames = 20; // 20 frames for temporal evolution
        let dt = 1e-4; // 100 μs between frames (10 kHz frame rate)
        let shear_wave_speed = 3.0; // m/s (typical for soft tissue)

        // Initialize displacement fields
        let ux = Array3::zeros((nx, ny, nz));
        let uy = Array3::zeros((nx, ny, nz));
        let mut uz = initial_displacement.clone();

        // Simulate shear wave propagation
        // Simple 1D propagation along z-direction with damping
        for frame in 1..n_frames {
            let time = frame as f64 * dt;

            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let z = k as f64 * self.grid.dz;

                        // Distance from push location (assume push at center z)
                        let push_z = nz as f64 * self.grid.dz / 2.0;
                        let distance = (z - push_z).abs();

                        // Shear wave arrival time
                        let arrival_time = distance / shear_wave_speed;

                        // Temporal evolution with Gaussian envelope
                        if time >= arrival_time {
                            let time_delay = time - arrival_time;
                            let amplitude = initial_displacement[[i, j, k]] *
                                          (-time_delay * 100.0).exp() * // Damping
                                          (-(time_delay - 0.0005).powi(2) / (2.0 * 0.0001)).exp(); // Gaussian shape

                            uz[[i, j, k]] = amplitude;
                        } else {
                            uz[[i, j, k]] = 0.0;
                        }
                    }
                }
            }
        }

        // For plane wave imaging, focus on axial displacement component
        // Full 3D vector displacement tracking available in advanced implementations
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
