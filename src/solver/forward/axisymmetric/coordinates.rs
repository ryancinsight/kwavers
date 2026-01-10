//! Cylindrical coordinate system for axisymmetric solver
//!
//! Handles the mapping between physical cylindrical coordinates (r, z)
//! and the 2D computational grid.

#![allow(deprecated)]

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Cylindrical grid for axisymmetric simulations
///
/// Maps physical (r, z) coordinates to 2D array indices where:
/// - First index (i) corresponds to axial position z
/// - Second index (j) corresponds to radial position r
#[derive(Debug, Clone)]
pub struct CylindricalGrid {
    /// Number of grid points in axial (z) direction
    pub nz: usize,
    /// Number of grid points in radial (r) direction  
    pub nr: usize,
    /// Grid spacing in axial direction (m)
    pub dz: f64,
    /// Grid spacing in radial direction (m)
    pub dr: f64,
    /// Axial coordinates (m)
    pub z: Array1<f64>,
    /// Radial coordinates (m)
    pub r: Array1<f64>,
    /// Axial wavenumbers (1/m)
    pub kz: Array1<f64>,
    /// Radial wavenumbers for Hankel transform (1/m)
    pub kr: Array1<f64>,
}

impl CylindricalGrid {
    /// Create a new cylindrical grid
    ///
    /// # Arguments
    ///
    /// * `nz` - Number of axial grid points
    /// * `nr` - Number of radial grid points
    /// * `dz` - Axial grid spacing (m)
    /// * `dr` - Radial grid spacing (m)
    pub fn new(nz: usize, nr: usize, dz: f64, dr: f64) -> KwaversResult<Self> {
        if nz == 0 || nr == 0 {
            return Err(crate::core::error::KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "grid dimensions".to_string(),
                    value: format!("nz={}, nr={}", nz, nr),
                    constraint: "Must be positive".to_string(),
                },
            ));
        }

        // Create spatial coordinates
        let z = Array1::from_shape_fn(nz, |i| i as f64 * dz);
        let r = Array1::from_shape_fn(nr, |j| j as f64 * dr);

        // Create axial wavenumbers (standard FFT frequencies)
        let kz = Self::compute_wavenumbers(nz, dz);

        // Create radial wavenumbers for Hankel transform
        // Using zeros of Bessel function J0 for discrete Hankel transform
        let kr = Self::compute_radial_wavenumbers(nr, dr);

        Ok(Self {
            nz,
            nr,
            dz,
            dr,
            z,
            r,
            kz,
            kr,
        })
    }

    /// Compute standard FFT wavenumbers
    fn compute_wavenumbers(n: usize, d: f64) -> Array1<f64> {
        let dk = 2.0 * PI / (n as f64 * d);
        Array1::from_shape_fn(n, |i| {
            if i <= n / 2 {
                i as f64 * dk
            } else {
                (i as f64 - n as f64) * dk
            }
        })
    }

    /// Compute radial wavenumbers for discrete Hankel transform
    ///
    /// Uses approximation for zeros of J0 Bessel function:
    /// j_{0,m} ≈ (m - 0.25) * π for large m
    fn compute_radial_wavenumbers(nr: usize, dr: f64) -> Array1<f64> {
        let r_max = nr as f64 * dr;
        Array1::from_shape_fn(nr, |m| {
            if m == 0 {
                0.0
            } else {
                // Approximate zeros of J0
                let j0m = (m as f64 - 0.25) * PI;
                j0m / r_max
            }
        })
    }

    /// Get physical axial coordinate for grid index
    #[inline]
    pub fn z_at(&self, i: usize) -> f64 {
        i as f64 * self.dz
    }

    /// Get physical radial coordinate for grid index
    #[inline]
    pub fn r_at(&self, j: usize) -> f64 {
        j as f64 * self.dr
    }

    /// Get grid index for axial coordinate (nearest)
    pub fn iz_for(&self, z: f64) -> usize {
        ((z / self.dz).round() as usize).min(self.nz - 1)
    }

    /// Get grid index for radial coordinate (nearest)
    pub fn ir_for(&self, r: f64) -> usize {
        ((r / self.dr).round() as usize).min(self.nr - 1)
    }

    /// Total number of grid points
    pub fn size(&self) -> usize {
        self.nz * self.nr
    }

    /// Maximum axial extent (m)
    pub fn z_max(&self) -> f64 {
        (self.nz - 1) as f64 * self.dz
    }

    /// Maximum radial extent (m)
    pub fn r_max(&self) -> f64 {
        (self.nr - 1) as f64 * self.dr
    }

    /// Create 2D meshgrid of z coordinates
    pub fn z_mesh(&self) -> Array2<f64> {
        let mut mesh = Array2::zeros((self.nz, self.nr));
        for i in 0..self.nz {
            let z = self.z_at(i);
            for j in 0..self.nr {
                mesh[[i, j]] = z;
            }
        }
        mesh
    }

    /// Create 2D meshgrid of r coordinates
    pub fn r_mesh(&self) -> Array2<f64> {
        let mut mesh = Array2::zeros((self.nz, self.nr));
        for i in 0..self.nz {
            for j in 0..self.nr {
                mesh[[i, j]] = self.r_at(j);
            }
        }
        mesh
    }

    /// Calculate area element for integration: r * dr * dz
    ///
    /// Note: At r=0, returns dr * dz (avoiding singularity)
    pub fn area_element(&self, j: usize) -> f64 {
        let r = self.r_at(j);
        if j == 0 {
            // Use half-cell width for r=0
            0.5 * self.dr * self.dr * self.dz
        } else {
            r * self.dr * self.dz
        }
    }

    /// Calculate volume of rotation for full 3D
    ///
    /// Returns 2π * r * dr * dz
    pub fn volume_element(&self, j: usize) -> f64 {
        2.0 * PI * self.area_element(j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn test_grid_creation() {
        let grid = CylindricalGrid::new(64, 32, 1e-4, 1e-4).unwrap();
        assert_eq!(grid.nz, 64);
        assert_eq!(grid.nr, 32);
        assert_eq!(grid.size(), 64 * 32);
    }

    #[test]
    #[allow(deprecated)]
    fn test_coordinates() {
        let grid = CylindricalGrid::new(10, 10, 0.1, 0.1).unwrap();

        assert_eq!(grid.z_at(0), 0.0);
        assert!((grid.z_at(5) - 0.5).abs() < 1e-10);

        assert_eq!(grid.r_at(0), 0.0);
        assert!((grid.r_at(3) - 0.3).abs() < 1e-10);
    }

    #[test]
    #[allow(deprecated)]
    fn test_index_lookup() {
        let grid = CylindricalGrid::new(100, 50, 1e-3, 1e-3).unwrap();

        assert_eq!(grid.iz_for(0.05), 50);
        assert_eq!(grid.ir_for(0.025), 25);
    }

    #[test]
    #[allow(deprecated)]
    fn test_wavenumbers() {
        let grid = CylindricalGrid::new(64, 32, 1e-4, 1e-4).unwrap();

        // DC component should be zero
        assert_eq!(grid.kz[0], 0.0);
        assert_eq!(grid.kr[0], 0.0);

        // Wavenumbers should be positive for first half
        assert!(grid.kz[1] > 0.0);
        assert!(grid.kr[1] > 0.0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_meshgrid() {
        let grid = CylindricalGrid::new(4, 3, 1.0, 1.0).unwrap();

        let z_mesh = grid.z_mesh();
        let r_mesh = grid.r_mesh();

        assert_eq!(z_mesh.dim(), (4, 3));
        assert_eq!(r_mesh.dim(), (4, 3));

        // Check values
        assert_eq!(z_mesh[[2, 0]], 2.0);
        assert_eq!(r_mesh[[0, 2]], 2.0);
    }
}
