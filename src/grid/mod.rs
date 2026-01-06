//! Grid module for spatial discretization
//!
//! This module provides the core grid structures and utilities for
//! defining computational domains and spatial discretization.

pub mod coordinates;
pub mod field_ops;
use crate::fft::kspace;
pub mod operators;
pub mod stability;
pub mod structure;
use crate::fft::utils as fft_utils;

// Re-exports for convenience
pub use coordinates::CoordinateSystem;
pub use fft_utils::*;
pub use kspace::KSpaceCalculator;
pub use structure::{Bounds, Dimension, Grid};

// Extension methods for Grid (compatibility layer)
impl Grid {
    /// Get minimum spacing
    #[inline]
    pub fn min_spacing(&self) -> f64 {
        self.dx.min(self.dy).min(self.dz)
    }

    /// Get maximum spacing
    #[inline]
    pub fn max_spacing(&self) -> f64 {
        self.dx.max(self.dy).max(self.dz)
    }

    /// Get physical dimensions of the domain
    #[inline]
    pub fn physical_size(&self) -> (f64, f64, f64) {
        (
            self.nx as f64 * self.dx,
            self.ny as f64 * self.dy,
            self.nz as f64 * self.dz,
        )
    }

    /// Get grid volume
    #[inline]
    pub fn volume(&self) -> f64 {
        let (lx, ly, lz) = self.physical_size();
        lx * ly * lz
    }

    /// Get cell volume
    #[inline]
    pub fn cell_volume(&self) -> f64 {
        self.dx * self.dy * self.dz
    }

    /// Check if a point is inside the grid
    #[inline]
    pub fn contains_point(&self, x: f64, y: f64, z: f64) -> bool {
        let (lx, ly, lz) = self.physical_size();
        x >= 0.0 && x <= lx && y >= 0.0 && y <= ly && z >= 0.0 && z <= lz
    }

    /// Get the bounds of the grid
    #[inline]
    pub fn bounds(&self) -> Bounds {
        let (lx, ly, lz) = self.physical_size();
        Bounds::new([0.0, 0.0, 0.0], [lx, ly, lz])
    }

    /// Convert coordinates to indices (nearest neighbor)
    #[inline]
    pub fn coordinates_to_indices(&self, x: f64, y: f64, z: f64) -> Option<(usize, usize, usize)> {
        if !self.contains_point(x, y, z) {
            return None;
        }
        Some((
            (x / self.dx) as usize,
            (y / self.dy) as usize,
            (z / self.dz) as usize,
        ))
    }

    /// Compute kx array (compatibility)
    #[inline]
    pub fn compute_kx(&self) -> ndarray::Array1<f64> {
        KSpaceCalculator::generate_kx(self)
    }

    /// Compute ky array (compatibility)
    #[inline]
    pub fn compute_ky(&self) -> ndarray::Array1<f64> {
        KSpaceCalculator::generate_ky(self)
    }

    /// Compute kz array (compatibility)
    #[inline]
    pub fn compute_kz(&self) -> ndarray::Array1<f64> {
        KSpaceCalculator::generate_kz(self)
    }

    /// Calculate CFL timestep for given sound speed
    ///
    /// Uses FDTD stability condition with safety factor
    #[inline]
    pub fn cfl_timestep(&self, max_sound_speed: f64) -> f64 {
        stability::StabilityCalculator::cfl_timestep_fdtd(self, max_sound_speed)
    }

    /// Get kx array (compatibility)
    #[inline]
    pub fn kx(&self) -> ndarray::Array1<f64> {
        KSpaceCalculator::generate_kx(self)
    }

    /// Get ky array (compatibility)
    #[inline]
    pub fn ky(&self) -> ndarray::Array1<f64> {
        KSpaceCalculator::generate_ky(self)
    }

    /// Get kz array (compatibility)
    #[inline]
    pub fn kz(&self) -> ndarray::Array1<f64> {
        KSpaceCalculator::generate_kz(self)
    }

    /// Get x coordinates (compatibility)
    #[inline]
    pub fn x_coordinates(&self) -> ndarray::Array1<f64> {
        CoordinateSystem::generate_x_vector(self)
    }

    /// Get y coordinates (compatibility)
    #[inline]
    pub fn y_coordinates(&self) -> ndarray::Array1<f64> {
        CoordinateSystem::generate_y_vector(self)
    }

    /// Get z coordinates (compatibility)
    #[inline]
    pub fn z_coordinates(&self) -> ndarray::Array1<f64> {
        CoordinateSystem::generate_z_vector(self)
    }

    /// Get maximum wavenumber supported by grid
    #[inline]
    pub fn k_max(&self) -> f64 {
        self.k_max
    }
}
