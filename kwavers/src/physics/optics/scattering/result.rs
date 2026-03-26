//! Mie scattering results containing efficiencies and cross-sections
//!
//! ## Efficiencies
//! - **Scattering Efficiency**: Q_sca = σ_sca / (πr²)
//! - **Extinction Efficiency**: Q_ext = σ_ext / (πr²)
//! - **Absorption Efficiency**: Q_abs = Q_ext - Q_sca

use std::f64::consts::PI;

/// Mie scattering results
#[derive(Debug, Clone)]
pub struct MieResult {
    /// Size parameter x
    pub size_parameter: f64,
    /// Scattering efficiency Q_sca
    pub scattering_efficiency: f64,
    /// Extinction efficiency Q_ext
    pub extinction_efficiency: f64,
    /// Absorption efficiency Q_abs
    pub absorption_efficiency: f64,
    /// Backscattering efficiency Q_bsa
    pub backscattering_efficiency: f64,
    /// Scattering cross-section \[m²\]
    pub scattering_cross_section: f64,
    /// Extinction cross-section \[m²\]
    pub extinction_cross_section: f64,
    /// Absorption cross-section \[m²\]
    pub absorption_cross_section: f64,
    /// Asymmetry parameter g (anisotropy)
    pub asymmetry_parameter: f64,
    /// Phase function at 180° (backscattering)
    pub phase_function_180: f64,
}

impl MieResult {
    /// Calculate geometric cross-section πr²
    #[must_use]
    pub fn geometric_cross_section(radius: f64) -> f64 {
        PI * radius * radius
    }

    /// Calculate albedo ω = Q_sca / Q_ext
    #[must_use]
    pub fn single_scatter_albedo(&self) -> f64 {
        if self.extinction_efficiency > 0.0 {
            self.scattering_efficiency / self.extinction_efficiency
        } else {
            0.0
        }
    }
}
