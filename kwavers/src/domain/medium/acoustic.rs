//! Acoustic properties trait for wave propagation
//!
//! This module defines traits for acoustic-specific properties including
//! absorption, attenuation, and nonlinear effects.

use crate::core::constants::acoustic_parameters::{WATER_ABSORPTION_ALPHA_0, WATER_NONLINEARITY_B_A};
use crate::domain::grid::Grid;
use crate::domain::medium::absorption::AbsorptionTissueType;
use crate::domain::medium::core::CoreMedium;

/// Trait for acoustic wave propagation properties
pub trait AcousticProperties: CoreMedium {
    /// Get absorption coefficient at a specific point and frequency (Np/m)
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64;

    /// Get acoustic attenuation coefficient (Np/m)
    fn attenuation(&self, _x: f64, _y: f64, _z: f64, frequency: f64, _grid: &Grid) -> f64 {
        // Default power-law absorption: α(f) = WATER_ABSORPTION_ALPHA_0 · f^1.05
        WATER_ABSORPTION_ALPHA_0 * frequency.powf(1.05)
    }

    /// Get nonlinearity parameter (B/A) for nonlinear wave propagation.
    ///
    /// Default: `WATER_NONLINEARITY_B_A = 5.0` (Beyer 1960; Duck 1990 Table 4.4).
    fn nonlinearity_parameter(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        WATER_NONLINEARITY_B_A
    }

    /// Get nonlinearity coefficient (beta) - alternative parameterization
    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Beta = 1 + B/(2A)
        1.0 + self.nonlinearity_parameter(x, y, z, grid) / 2.0
    }

    /// Get the raw power-law absorption prefactor α₀ at 1 MHz [dB/(MHz^y·cm)].
    ///
    /// The spectral solvers convert this coefficient to
    /// `Np/((rad/s)^y·m)` at the solver boundary before forming `τ` and `η`.
    /// Default implementation returns 0 if the medium does not expose an
    /// explicit power-law coefficient.
    fn alpha_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.0
    }

    /// Get the power-law absorption exponent `y`.
    ///
    /// Default is 1.05, which matches the water reference medium used by the
    /// homogeneous implementation.
    fn alpha_power(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.05
    }

    /// Get the tissue type at a specific position (if medium supports tissue types)
    fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<AbsorptionTissueType> {
        None
    }

    /// Get acoustic diffusivity (m²/s)
    /// Represents the ratio of thermal diffusivity to acoustic velocity
    fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
}
