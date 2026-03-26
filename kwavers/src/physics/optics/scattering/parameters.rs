//! Mie scattering parameters for a spherical particle
//!
//! ## Key Parameters
//! - **Size Parameter**: x = 2πr/λ
//! - **Refractive Index**: m = n_particle / n_medium

use std::f64::consts::PI;

/// Mie scattering parameters for a spherical particle
#[derive(Debug, Clone)]
pub struct MieParameters {
    /// Particle radius \[m\]
    pub radius: f64,
    /// Particle refractive index (complex)
    pub refractive_index: num_complex::Complex64,
    /// Medium refractive index (real)
    pub medium_index: f64,
    /// Wavelength in medium \[m\]
    pub wavelength: f64,
}

impl MieParameters {
    /// Create new Mie parameters
    #[must_use]
    pub fn new(
        radius: f64,
        refractive_index: num_complex::Complex64,
        medium_index: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            radius,
            refractive_index,
            medium_index,
            wavelength,
        }
    }

    /// Calculate size parameter x = 2πr/λ
    #[must_use]
    pub fn size_parameter(&self) -> f64 {
        2.0 * PI * self.radius * self.medium_index / self.wavelength
    }

    /// Calculate relative refractive index m = n_particle / n_medium
    #[must_use]
    pub fn relative_index(&self) -> num_complex::Complex64 {
        self.refractive_index / self.medium_index
    }
}
