//! k-Wave Configuration Types
//!
//! This module contains all configuration structures and enums for k-Wave compatibility,
//! following the Single Responsibility Principle (GRASP).

use ndarray::Array3;

/// k-Wave simulation configuration matching MATLAB interface
#[derive(Debug, Clone)]
pub struct KWaveConfig {
    /// Enable power law absorption
    pub absorption_mode: AbsorptionMode,
    /// Enable nonlinear acoustics
    pub nonlinearity: bool,
    /// PML size in grid points
    pub pml_size: usize,
    /// PML absorption coefficient
    pub pml_alpha: f64,
    /// Data recording options
    pub sensor_mask: Option<Array3<bool>>,
    /// Perfectly matched layer inside domain
    pub pml_inside: bool,
    /// Smooth source terms
    pub smooth_sources: bool,
}

/// Absorption models supported by k-Wave
#[derive(Debug, Clone, Copy)]
pub enum AbsorptionMode {
    /// No absorption
    Lossless,
    /// Stokes absorption (frequency squared)
    Stokes,
    /// Power law absorption: α = α₀ω^y
    PowerLaw { alpha_coeff: f64, alpha_power: f64 },
}

impl Default for KWaveConfig {
    fn default() -> Self {
        Self {
            absorption_mode: AbsorptionMode::Lossless,
            nonlinearity: false,
            pml_size: 20,
            pml_alpha: 2.0,
            sensor_mask: None,
            pml_inside: true,
            smooth_sources: true,
        }
    }
}