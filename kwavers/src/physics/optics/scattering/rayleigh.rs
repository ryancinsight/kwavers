//! Simplified Mie scattering for small particles (Rayleigh limit)

use std::f64::consts::PI;

/// Simplified Mie scattering for small particles (Rayleigh limit)
#[derive(Debug)]
pub struct RayleighScattering {
    /// Wavelength \[m\]
    pub wavelength: f64,
    /// Particle radius \[m\]
    pub radius: f64,
    /// |α|² — squared magnitude of polarizability \[m⁶\]
    pub polarizability: f64,
    /// Im(α) — imaginary part of polarizability, drives absorption \[m³\]
    polarizability_imag: f64,
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
            polarizability_imag: alpha_complex.im,
        }
    }

    /// Calculate Rayleigh scattering cross-section
    ///
    /// σ_scat = (8π/3) k⁴ |α|²  where k = 2π/λ
    #[must_use]
    pub fn scattering_cross_section(&self) -> f64 {
        let k = 2.0 * PI / self.wavelength;
        (8.0 * PI / 3.0) * k.powi(4) * self.polarizability
    }

    /// Calculate Rayleigh extinction cross-section
    ///
    /// σ_ext = σ_scat + σ_abs = (8π/3) k⁴ |α|² + 4π k Im(α)
    #[must_use]
    pub fn extinction_cross_section(&self) -> f64 {
        let k = 2.0 * PI / self.wavelength;
        let sigma_scat = (8.0 * PI / 3.0) * k.powi(4) * self.polarizability;
        let sigma_abs = 4.0 * PI * k * self.polarizability_imag;
        sigma_scat + sigma_abs
    }

    /// Calculate depolarization factor
    #[must_use]
    pub fn depolarization_factor(&self) -> f64 {
        // For spherical particles, ρ = 0 (no depolarization)
        0.0
    }
}
