//! Mie theory implementation for spherical plasmonic nanoparticles
//!
//! Describes scattering and absorption of electromagnetic radiation by spherical
//! particles.

use std::f64::consts::PI;

/// Mie theory calculator for spherical plasmonic nanoparticles
pub struct MieTheory {
    /// Nanoparticle radius (m)
    pub radius: f64,
    /// Dielectric function of nanoparticle ε_particle(ω)
    pub particle_dielectric: Box<dyn Fn(f64) -> num_complex::Complex<f64>>,
    /// Dielectric function of surrounding medium ε_medium
    pub medium_dielectric: f64,
}

impl std::fmt::Debug for MieTheory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MieTheory")
            .field("radius", &self.radius)
            .field("medium_dielectric", &self.medium_dielectric)
            .finish()
    }
}

impl MieTheory {
    /// Create Mie theory calculator for gold nanoparticle in water
    ///
    /// Uses a simplified Drude-Lorentz optical property model for gold.
    #[must_use]
    pub fn gold_in_water(radius: f64) -> Self {
        Self {
            radius,
            particle_dielectric: Box::new(|wavelength| {
                // Simplified Drude-Lorentz model for gold
                // ε(ω) = ε_inf - ω_p²/(ω² + iγω)

                let epsilon_inf = 9.84;
                let omega_p = 9.01; // eV
                let gamma = 0.071; // eV

                // Convert wavelength to energy (E = hc/λ)
                let hbar = 6.582119569e-16; // eV·s
                let energy = 1.23984193e-6 / wavelength; // Convert m to eV

                let omega = energy / hbar;
                let omega_p_rad = omega_p / hbar;
                let gamma_rad = gamma / hbar;

                let denominator = num_complex::Complex::new(omega * omega, gamma_rad * omega);
                epsilon_inf - (omega_p_rad * omega_p_rad) / denominator
            }),
            // Water at optical frequencies
            medium_dielectric: 1.77,
        }
    }

    /// Compute polarizability (α) using Mie theory
    #[must_use]
    pub fn polarizability(&self, wavelength: f64) -> num_complex::Complex<f64> {
        let eps_particle = (self.particle_dielectric)(wavelength);
        let eps_medium = self.medium_dielectric;

        // Mie polarizability (quasistatic limit): α = 4π ε₀ ε_m R³ (ε - ε_m)/(ε + 2ε_m)
        let eps_ratio = eps_particle / eps_medium;
        let numerator = eps_ratio - num_complex::Complex::new(1.0, 0.0);
        let denominator = eps_ratio + num_complex::Complex::new(2.0, 0.0);

        let alpha_dimensionless =
            3.0 * self.radius * self.radius * self.radius * numerator / denominator;

        // Convert to SI units (include 4π ε₀ ε_m factor)
        let epsilon0 = 8.854e-12;
        alpha_dimensionless * 4.0 * PI * epsilon0 * eps_medium
    }

    /// Compute scattering cross-section (σ_scat)
    #[must_use]
    pub fn scattering_cross_section(&self, wavelength: f64) -> f64 {
        let alpha = self.polarizability(wavelength);
        let k = 2.0 * PI / wavelength; // Wave number in medium

        // σ_scat = (8π/3) k⁴ |α|² for Rayleigh scattering
        (8.0 * PI / 3.0) * k.powi(4) * alpha.norm_sqr()
    }

    /// Compute absorption cross-section (σ_abs)
    #[must_use]
    pub fn absorption_cross_section(&self, wavelength: f64) -> f64 {
        let alpha = self.polarizability(wavelength);
        let k = 2.0 * PI / wavelength;

        // σ_abs = k Im(α) for small particles
        k * alpha.im
    }

    /// Compute extinction cross-section (σ_ext = σ_scat + σ_abs)
    #[must_use]
    pub fn extinction_cross_section(&self, wavelength: f64) -> f64 {
        self.scattering_cross_section(wavelength) + self.absorption_cross_section(wavelength)
    }

    /// Find plasmon resonance wavelength by minimizing the denominator Re(ε_particle + 2ε_medium)
    #[must_use]
    pub fn plasmon_resonance_wavelength(&self) -> Option<f64> {
        // Simple grid search
        let wavelengths = (400..900).map(|nm| f64::from(nm) * 1e-9); // 400-900 nm

        for wavelength in wavelengths {
            let eps_particle = (self.particle_dielectric)(wavelength);
            let denominator =
                eps_particle + num_complex::Complex::new(2.0 * self.medium_dielectric, 0.0);

            if denominator.re.abs() < 0.1 {
                // Close to resonance
                return Some(wavelength);
            }
        }

        None
    }
}
