//! Grid module for spatial discretization
//!
//! This module provides a domain-driven structure for grid operations:
//! - `structure`: Core grid definition and properties
//! - `coordinates`: Coordinate generation and position conversions
//! - `kspace`: K-space operations for spectral methods
//! - `field_ops`: Field array creation and manipulation
//! - `stability`: Numerical stability and CFL conditions

pub mod coordinates;
pub mod field_ops;
pub mod kspace;
pub mod stability;
pub mod structure;

// Re-export main types for convenience
pub use coordinates::CoordinateSystem;
pub use field_ops::{FieldOperations, FieldStatistics};
pub use kspace::KSpaceCalculator;
pub use stability::StabilityCalculator;
pub use structure::{Bounds, Dimension, Grid};

// Extension methods for Grid to provide convenient access and compatibility
impl Grid {
    /// Create a zero-initialized field
    #[inline]
    pub fn create_field(&self) -> ndarray::Array3<f64> {
        FieldOperations::create_field(self)
    }

    /// Convert position to indices
    #[inline]
    pub fn position_to_indices(&self, x: f64, y: f64, z: f64) -> Option<(usize, usize, usize)> {
        CoordinateSystem::position_to_indices(self, x, y, z)
    }

    /// Get CFL timestep
    #[inline]
    pub fn cfl_timestep(&self, max_sound_speed: f64) -> f64 {
        StabilityCalculator::cfl_timestep_fdtd(self, max_sound_speed)
    }

    /// Get CFL timestep with default safety factor (compatibility)
    #[inline]
    pub fn cfl_timestep_default(&self, max_sound_speed: f64) -> f64 {
        StabilityCalculator::cfl_timestep_fdtd(self, max_sound_speed)
    }

    /// Get coordinates for a dimension (compatibility)
    #[inline]
    pub fn coordinates(&self, dim: Dimension) -> ndarray::Array1<f64> {
        CoordinateSystem::generate_coordinate_vector(self, dim)
    }

    /// Get k-squared array (compatibility)
    #[inline]
    pub fn k_squared(&self) -> &ndarray::Array3<f64> {
        KSpaceCalculator::get_k_squared_cached(self)
    }

    /// Generate k-space arrays (compatibility)
    #[inline]
    pub fn generate_k(
        &self,
    ) -> (
        ndarray::Array1<f64>,
        ndarray::Array1<f64>,
        ndarray::Array1<f64>,
    ) {
        (
            KSpaceCalculator::generate_kx(self),
            KSpaceCalculator::generate_ky(self),
            KSpaceCalculator::generate_kz(self),
        )
    }

    /// Convert indices to coordinates (compatibility - overloaded method)
    #[inline]
    pub fn indices_to_coordinates(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        (i as f64 * self.dx, j as f64 * self.dy, k as f64 * self.dz)
    }

    /// Get grid dimensions (compatibility)
    #[inline]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
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

    /// Get grid spacing (compatibility)
    #[inline]
    pub fn spacing(&self) -> (f64, f64, f64) {
        (self.dx, self.dy, self.dz)
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
}
