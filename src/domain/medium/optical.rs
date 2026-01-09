//! Optical properties trait for light propagation
//!
//! This module defines traits for optical properties including absorption,
//! scattering, and refractive index.

use crate::domain::grid::Grid;
use crate::domain::medium::core::CoreMedium;

/// Trait for optical medium properties
pub trait OpticalProperties: CoreMedium {
    /// Get optical absorption coefficient (1/m)
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get optical scattering coefficient (1/m)
    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get refractive index
    fn refractive_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.33 // Default for water
    }

    /// Get anisotropy factor for scattering (g parameter in Henyey-Greenstein)
    fn anisotropy_factor(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.9 // Default for tissue
    }

    /// Get reduced scattering coefficient μ'_s = μ_s(1-g) (1/m)
    fn reduced_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu_s = self.optical_scattering_coefficient(x, y, z, grid);
        let g = self.anisotropy_factor(x, y, z, grid);
        mu_s * (1.0 - g)
    }
}
