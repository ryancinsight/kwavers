//! Acoustic properties trait for wave propagation
//!
//! This module defines traits for acoustic-specific properties including
//! absorption, attenuation, and nonlinear effects.

use crate::absorption::AbsorptionTissueType;
use crate::core::CoreMedium;
use aequitas::systems::si::quantities::{Dimensionless, ReciprocalLength};
use kwavers_core::constants::acoustic_parameters::{
    DB_TO_NP, WATER_ABSORPTION_ALPHA_0, WATER_ABSORPTION_POWER, WATER_NONLINEARITY_B_A,
};
use kwavers_grid::Grid;

/// Trait for acoustic wave propagation properties
pub trait AcousticProperties: CoreMedium {
    /// Get absorption coefficient at a specific point and frequency
    fn absorption_coefficient(
        &self,
        x: f64,
        y: f64,
        z: f64,
        grid: &Grid,
        frequency: f64,
    ) -> ReciprocalLength<f64>;

    /// Get acoustic attenuation coefficient
    fn attenuation(
        &self,
        _x: f64,
        _y: f64,
        _z: f64,
        frequency: f64,
        _grid: &Grid,
    ) -> ReciprocalLength<f64> {
        // Water power-law absorption (Szabo 1994): the stored prefactor
        // WATER_ABSORPTION_ALPHA_0 is in dB/(cm·MHz^y) with y = WATER_ABSORPTION_POWER = 2
        // (Stokes–Kirchhoff f² law). Convert dB/cm at `frequency` [Hz] to Np/m:
        // × 100 [cm/m] × DB_TO_NP [Np/dB]. At 1 MHz this gives 0.0253 Np/m.
        let alpha_db_per_cm = WATER_ABSORPTION_ALPHA_0
            * (frequency / kwavers_core::constants::MHZ_TO_HZ).powf(WATER_ABSORPTION_POWER);
        ReciprocalLength::from_base(alpha_db_per_cm * DB_TO_NP * 100.0)
    }

    /// Get nonlinearity parameter (B/A) for nonlinear wave propagation.
    ///
    /// Default: `WATER_NONLINEARITY_B_A = 5.0` (Beyer 1960; Duck 1990 Table 4.4).
    fn nonlinearity_parameter(
        &self,
        _x: f64,
        _y: f64,
        _z: f64,
        _grid: &Grid,
    ) -> Dimensionless<f64> {
        Dimensionless::from_base(WATER_NONLINEARITY_B_A)
    }

    /// Get nonlinearity coefficient (beta) - alternative parameterization
    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Dimensionless<f64> {
        // Beta = 1 + B/(2A)
        Dimensionless::from_base(1.0 + self.nonlinearity_parameter(x, y, z, grid).into_base() / 2.0)
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

    // Sound diffusivity δ [m²/s] is deliberately not a trait getter: it is
    // frequency-derived, δ = 2αc³/ω² (Hamilton & Blackstock), and is owned by
    // kwavers-physics `compute_diffusivity_from_power_law_absorption`.
}
