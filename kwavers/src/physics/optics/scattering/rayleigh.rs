//! Simplified Mie scattering for small particles (Rayleigh limit)

use std::f64::consts::PI;

/// Simplified Mie scattering for small particles (Rayleigh limit)
#[derive(Debug)]
pub struct RayleighScattering {
    /// Wavelength \[m\]
    pub wavelength: f64,
    /// Particle radius \[m\]
    pub radius: f64,
    /// Particle polarizability \[m³\]
    pub polarizability: f64,
}

impl RayleighScattering {
    /// Create Rayleigh scattering calculator
    #[must_use]
    pub fn new(wavelength: f64, radius: f64, refractive_index: num_complex::Complex64) -> Self {
        // Calculate polarizability using Lorentz-Lorenz relation
        // α = 4πϵ₀ r³ (m²-1)/(m²+2) where m = n + ik
        let m = refractive_index;
        let m2 = m * m;
        let alpha_complex = 4.0 * PI * radius.powi(3) * (m2 - 1.0) / (m2 + 2.0);
        Self {
            wavelength,
            radius,
            polarizability: alpha_complex.norm_sqr(),
        }
    }

    /// Calculate Rayleigh scattering cross-section
    #[must_use]
    pub fn scattering_cross_section(&self) -> f64 {
        (8.0 / 3.0) * PI.powi(4) * self.polarizability / self.wavelength.powi(4)
    }

    /// Calculate Rayleigh extinction cross-section
    #[must_use]
    pub fn extinction_cross_section(&self) -> f64 {
        (8.0 * PI.powi(2) / 3.0) * self.polarizability / self.wavelength.powi(2)
    }

    /// Calculate depolarization factor
    #[must_use]
    pub fn depolarization_factor(&self) -> f64 {
        // For spherical particles, ρ = 0 (no depolarization)
        0.0
    }
}
