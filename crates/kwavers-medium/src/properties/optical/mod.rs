//! Optical material property data structures for light propagation and scattering
//!
//! # Mathematical Foundation
//!
//! ## Radiative Transfer Equation (RTE)
//!
//! Light propagation in scattering media:
//! ```text
//! dI/ds = -μ_t I + μ_s ∫ p(θ) I(s') dΩ'
//! ```
//!
//! Where:
//! - `I`: Radiance (W/m²/sr)
//! - `s`: Path length (m)
//! - `μ_t = μ_a + μ_s`: Total attenuation coefficient (m⁻¹)
//! - `μ_a`: Absorption coefficient (m⁻¹)
//! - `μ_s`: Scattering coefficient (m⁻¹)
//! - `p(θ)`: Phase function (angular scattering probability)
//!
//! ## Henyey-Greenstein Phase Function
//!
//! Anisotropic scattering model:
//! ```text
//! p(θ) = (1 - g²) / (4π (1 + g² - 2g cos θ)^(3/2))
//! ```
//! - `g`: Anisotropy factor (⟨cos θ⟩)
//! - `g = 0`: Isotropic scattering
//! - `g > 0`: Forward scattering (typical for biological tissue)
//! - `g < 0`: Backward scattering
//!
//! ## Invariants
//!
//! - `absorption_coefficient ≥ 0` (m⁻¹)
//! - `scattering_coefficient ≥ 0` (m⁻¹)
//! - `-1 ≤ anisotropy ≤ 1` (dimensionless)
//! - `refractive_index ≥ 1.0` (vacuum is lower bound)

mod computed;
mod presets;
#[cfg(test)]
mod tests;

use std::fmt;

/// Canonical optical material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalPropertyData {
    /// Absorption coefficient μ_a (m⁻¹)
    pub absorption_coefficient: f64,

    /// Scattering coefficient μ_s (m⁻¹)
    pub scattering_coefficient: f64,

    /// Anisotropy factor g = ⟨cos θ⟩ (dimensionless)
    ///
    /// Physical range for biological tissue: 0.7-0.99 (highly forward-scattering)
    pub anisotropy: f64,

    /// Refractive index n (dimensionless)
    ///
    /// Ratio of light speed in vacuum to light speed in medium: n = c₀/c
    pub refractive_index: f64,
}

impl fmt::Display for OpticalPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Optical(μ_a={:.2} m⁻¹, μ_s={:.1} m⁻¹, μ_s'={:.1} m⁻¹, g={:.3}, n={:.3}, δ={:.1} mm, α={:.3})",
            self.absorption_coefficient,
            self.scattering_coefficient,
            self.reduced_scattering(),
            self.anisotropy,
            self.refractive_index,
            self.penetration_depth() * 1000.0,
            self.albedo()
        )
    }
}
