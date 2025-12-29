//! Acoustic properties trait for wave propagation
//!
//! This module defines traits for acoustic-specific properties including
//! absorption, attenuation, and nonlinear effects.

use crate::grid::Grid;
use crate::medium::absorption::TissueType;
use crate::medium::core::CoreMedium;

/// Trait for acoustic wave propagation properties
pub trait AcousticProperties: CoreMedium {
    /// Get absorption coefficient at a specific point and frequency (Np/m)
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64;

    /// Get acoustic attenuation coefficient (Np/m)
    fn attenuation(&self, _x: f64, _y: f64, _z: f64, frequency: f64, _grid: &Grid) -> f64 {
        // Default power law absorption for water
        0.0022 * frequency.powf(1.05)
    }

    /// Get nonlinearity parameter (B/A) for nonlinear wave propagation
    fn nonlinearity_parameter(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        5.0 // Default B/A for water-like media
    }

    /// Get nonlinearity coefficient (beta) - alternative parameterization
    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Beta = 1 + B/(2A)
        1.0 + self.nonlinearity_parameter(x, y, z, grid) / 2.0
    }

    /// Get absorption coefficient alpha_0 (dB/(MHz^y cm) or Np/(MHz^y m))
    /// This is the prefactor for power law absorption: alpha(f) = alpha_0 * f^y
    /// Default implementation assumes it's embedded in the attenuation function, returning 0 if not explicit.
    fn alpha_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.0
    }

    /// Get absorption power exponent y
    /// Default is 1.05 (typical for soft tissue)
    fn alpha_power(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.05
    }

    /// Get the tissue type at a specific position (if medium supports tissue types)
    fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<TissueType> {
        None
    }

    /// Get acoustic diffusivity (m²/s)
    /// Represents the ratio of thermal diffusivity to acoustic velocity
    fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
}
