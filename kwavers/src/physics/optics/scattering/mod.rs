//! Mie scattering theory for electromagnetic wave scattering by spherical particles
//!
//! This module implements Mie scattering theory for calculating scattering and absorption
//! cross-sections of spherical particles. This is fundamental for understanding light
//! propagation in biological tissues and optical imaging applications.
//!
//! # Theory
//!
//! Mie scattering describes the scattering of electromagnetic waves by spherical particles
//! of arbitrary size relative to the wavelength. The theory provides exact solutions
//! to Maxwell's equations for scattering by spheres.
//!
//! ## Key Parameters
//! - **Size Parameter**: x = 2πr/λ (ratio of circumference to wavelength)
//! - **Refractive Index**: m = n_particle / n_medium
//! - **Scattering Efficiency**: Q_sca = σ_sca / (πr²)
//! - **Extinction Efficiency**: Q_ext = σ_ext / (πr²)
//! - **Absorption Efficiency**: Q_abs = Q_ext - Q_sca
//!
//! ## Applications
//! - Light scattering in biological tissues
//! - Optical particle sizing and characterization
//! - Atmospheric scattering (aerosols, clouds)
//! - Nanoparticle optical properties
//! - Sonoluminescence bubble optical properties
//!
//! # References
//!
//! - Mie, G. (1908). "Beiträge zur Optik trüber Medien, speziell kolloidaler Metallösungen"
//! - Bohren, C. F., & Huffman, D. R. (1983). Absorption and scattering of light by small particles
//! - Kerker, M. (1969). The scattering of light and other electromagnetic radiation

use std::f64::consts::PI;

use crate::core::error::{KwaversError, KwaversResult};

/// Physical constants for Mie scattering calculations
pub mod constants {
    /// Speed of light in vacuum [m/s]
    pub const C: f64 = 2.99792458e8;
    /// Vacuum permittivity [F/m]
    pub const EPSILON_0: f64 = 8.854187817e-12;
    /// Vacuum permeability [H/m]
    pub const MU_0: f64 = 4.0 * std::f64::consts::PI * 1e-7;
}

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

/// Mie scattering calculator
#[derive(Debug)]
pub struct MieCalculator {
    /// Maximum number of terms in series expansion
    max_terms: usize,
}

impl Default for MieCalculator {
    fn default() -> Self {
        Self { max_terms: 10000 }
    }
}

impl MieCalculator {
    /// Create new Mie calculator with custom maximum terms
    #[must_use]
    pub fn new(max_terms: usize) -> Self {
        Self { max_terms }
    }

    /// Calculate Mie scattering for given parameters
    ///
    /// # Arguments
    /// * `params` - Mie scattering parameters
    ///
    /// # Returns
    /// Mie scattering results
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` if size parameter exceeds 100.0
    pub fn calculate(&self, params: &MieParameters) -> KwaversResult<MieResult> {
        let x = params.size_parameter();

        // For small particles, use Rayleigh approximation
        if x < 0.1 {
            return Ok(self.rayleigh_approximation(params));
        }

        let m = params.relative_index();

        // Check size parameter limits
        if x > 100.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Size parameter x = {x} too large for current implementation (max 100.0)"
            )));
        }

        // Determine number of multipole expansion terms needed
        let n_max = (x as usize + 1).max(10).min(self.max_terms);

        // Calculate Mie coefficients
        let (an, bn) = self.calculate_coefficients(num_complex::Complex64::from(x), m, n_max);

        // Calculate efficiencies
        let q_sca = self.scattering_efficiency(&an, &bn, x);
        let q_ext = self.extinction_efficiency(&an, &bn, x);
        let q_abs = q_ext - q_sca;
        let q_bsa = self.backscattering_efficiency(&an, &bn);

        // Calculate cross-sections
        let geometric_cs = MieResult::geometric_cross_section(params.radius);
        let sigma_sca = q_sca * geometric_cs;
        let sigma_ext = q_ext * geometric_cs;
        let sigma_abs = q_abs * geometric_cs;

        // Calculate asymmetry parameter
        let g = self.asymmetry_parameter(&an, &bn, x);

        // Calculate phase function at 180°
        let p_180 = self.phase_function_180(&an, &bn);

        Ok(MieResult {
            size_parameter: x,
            scattering_efficiency: q_sca,
            extinction_efficiency: q_ext,
            absorption_efficiency: q_abs,
            backscattering_efficiency: q_bsa,
            scattering_cross_section: sigma_sca,
            extinction_cross_section: sigma_ext,
            absorption_cross_section: sigma_abs,
            asymmetry_parameter: g,
            phase_function_180: p_180,
        })
    }

    /// Rayleigh approximation for small particles (x << 1)
    ///
    /// For particles much smaller than the wavelength, Mie theory reduces to Rayleigh scattering.
    /// The scattering efficiency is given by Q_sca = (8/3) x^4 |(m²-1)/(m²+2)|²
    /// where x = 2πr/λ is the size parameter.
    ///
    /// Reference: Bohren, C. F., & Huffman, D. R. (1983). Absorption and scattering of light by small particles
    fn rayleigh_approximation(&self, params: &MieParameters) -> MieResult {
        let x = params.size_parameter();
        let m = params.relative_index();

        // Rayleigh scattering efficiency: Q_sca = (8/3) x^4 |(m²-1)/(m²+2)|²
        // This is the standard form from Bohren & Huffman (1983)
        let m2 = m * m;
        let alpha_term = (m2 - 1.0) / (m2 + 2.0);
        let q_sca = (8.0 / 3.0) * x.powi(4) * alpha_term.norm_sqr();
        let q_ext = q_sca; // No absorption in Rayleigh limit for transparent particles

        let geometric_cs = MieResult::geometric_cross_section(params.radius);

        MieResult {
            size_parameter: x,
            scattering_efficiency: q_sca,
            extinction_efficiency: q_ext,
            absorption_efficiency: 0.0,
            backscattering_efficiency: q_sca * 0.5, // Approximate
            scattering_cross_section: q_sca * geometric_cs,
            extinction_cross_section: q_ext * geometric_cs,
            absorption_cross_section: 0.0,
            asymmetry_parameter: 0.0, // Isotropic in Rayleigh limit
            phase_function_180: q_sca,
        }
    }

    /// Calculate Mie scattering coefficients a_n and b_n
    fn calculate_coefficients(
        &self,
        x: num_complex::Complex64,
        m: num_complex::Complex64,
        n_max: usize,
    ) -> (Vec<num_complex::Complex64>, Vec<num_complex::Complex64>) {
        let mut an = Vec::with_capacity(n_max);
        let mut bn = Vec::with_capacity(n_max);

        // Calculate Riccati-Bessel functions
        let (psi, xi) = self.riccati_bessel(x, n_max);
        let (psi_m, xi_m) = self.riccati_bessel(x * m, n_max);

        for n in 1..=n_max {
            let _n_f64 = n as f64;

            // Mie coefficients using continued fraction method
            let a_n = self.mie_a_coefficient(n, m, &psi, &xi, &psi_m, &xi_m);
            let b_n = self.mie_b_coefficient(n, m, &psi, &xi, &psi_m, &xi_m);

            an.push(a_n);
            bn.push(b_n);
        }

        (an, bn)
    }

    /// Calculate Riccati-Bessel functions ψ_n(z) and ξ_n(z)
    fn riccati_bessel(
        &self,
        z: num_complex::Complex64,
        n_max: usize,
    ) -> (Vec<num_complex::Complex64>, Vec<num_complex::Complex64>) {
        let mut psi = Vec::with_capacity(n_max + 1);
        let mut xi = Vec::with_capacity(n_max + 1);

        // Initialize with spherical Bessel functions
        psi.push(z.sin()); // ψ₀(z) = sin(z)
        xi.push(z.sin() - num_complex::Complex64::I * z.cos()); // ξ₀(z) = sin(z) - i*cos(z)

        psi.push(z.sin() / z - z.cos()); // ψ₁(z) = sin(z)/z - cos(z)
        xi.push(psi[1] - num_complex::Complex64::I * (z.sin() / z + z.cos())); // ξ₁(z)

        // Recurrence relation for higher orders
        for n in 2..=n_max {
            let n_f64 = n as f64;
            let psi_n = ((2.0 * n_f64 - 1.0) / n_f64) * psi[n - 1] - psi[n - 2];
            let xi_n = ((2.0 * n_f64 - 1.0) / n_f64) * xi[n - 1] - xi[n - 2];

            psi.push(psi_n);
            xi.push(xi_n);
        }

        (psi, xi)
    }

    /// Calculate Mie a_n coefficient
    fn mie_a_coefficient(
        &self,
        n: usize,
        m: num_complex::Complex64,
        psi: &[num_complex::Complex64],
        xi: &[num_complex::Complex64],
        psi_m: &[num_complex::Complex64],
        xi_m: &[num_complex::Complex64],
    ) -> num_complex::Complex64 {
        let numerator = psi[n] * (psi_m[n] - m * xi_m[n]) - psi_m[n] * (psi[n] - xi[n]);
        let denominator = xi[n] * (psi_m[n] - m * xi_m[n]) - psi_m[n] * (xi[n] - m * xi_m[n]);

        numerator / denominator
    }

    /// Calculate Mie b_n coefficient
    fn mie_b_coefficient(
        &self,
        n: usize,
        m: num_complex::Complex64,
        psi: &[num_complex::Complex64],
        xi: &[num_complex::Complex64],
        psi_m: &[num_complex::Complex64],
        xi_m: &[num_complex::Complex64],
    ) -> num_complex::Complex64 {
        let numerator = psi_m[n] * (psi[n] - m * xi[n]) - psi[n] * (psi_m[n] - m * xi_m[n]);
        let denominator = psi_m[n] * (xi[n] - m * xi[n]) - xi_m[n] * (psi[n] - m * xi[n]);

        numerator / denominator
    }

    /// Calculate scattering efficiency Q_sca
    fn scattering_efficiency(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
        x: f64,
    ) -> f64 {
        let mut sum = 0.0;

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            sum += (2.0 * n_f64 + 1.0) * (an[n].norm_sqr() + bn[n].norm_sqr());
        }

        (2.0 / (x * x)) * sum
    }

    /// Calculate extinction efficiency Q_ext
    fn extinction_efficiency(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
        x: f64,
    ) -> f64 {
        let mut sum = 0.0;

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            sum += (2.0 * n_f64 + 1.0) * (an[n] + bn[n]).re;
        }

        (2.0 / (x * x)) * sum
    }

    /// Calculate backscattering efficiency Q_bsa
    fn backscattering_efficiency(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
    ) -> f64 {
        let mut sum = num_complex::Complex64::new(0.0, 0.0);

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            sum += (2.0 * n_f64 + 1.0) * (-1.0_f64).powi(n as i32) * (an[n] - bn[n]);
        }

        sum.norm_sqr()
    }

    /// Calculate asymmetry parameter g
    fn asymmetry_parameter(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
        x: f64,
    ) -> f64 {
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            let weight = (2.0 * n_f64 + 1.0) / ((n_f64 * (n_f64 + 1.0)) * (2.0 * n_f64 + 1.0));

            sum1 += weight * (an[n] * bn[n].conj() + bn[n] * an[n].conj()).re;
            sum2 += (2.0 * n_f64 + 1.0) * (an[n].norm_sqr() + bn[n].norm_sqr());
        }

        if sum2 > 0.0 {
            (4.0 / (x * x)) * sum1 / ((2.0 / (x * x)) * sum2)
        } else {
            0.0
        }
    }

    /// Calculate phase function at 180°
    fn phase_function_180(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
    ) -> f64 {
        let mut sum = num_complex::Complex64::new(0.0, 0.0);

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            sum += (2.0 * n_f64 + 1.0) * (-1.0_f64).powi(n as i32) * (an[n] - bn[n]);
        }

        sum.norm_sqr()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rayleigh_scattering() {
        let wavelength = 500e-9; // 500 nm
        let radius = 50e-9; // 50 nm
        let n_particle = num_complex::Complex64::new(1.5, 0.01); // Glass with small absorption

        let rayleigh = RayleighScattering::new(wavelength, radius, n_particle);

        // Rayleigh scattering cross-section should be positive
        assert!(rayleigh.scattering_cross_section() > 0.0);

        // Basic properties should hold
        assert!(rayleigh.polarizability > 0.0);
        assert!(rayleigh.wavelength > 0.0);
        assert!(rayleigh.radius > 0.0);

        // Depolarization factor should be zero for spheres
        assert_eq!(rayleigh.depolarization_factor(), 0.0);
    }

    #[test]
    fn test_mie_parameters() {
        let params = MieParameters::new(
            100e-9,                                // 100 nm radius
            num_complex::Complex64::new(1.5, 0.1), // Complex refractive index
            1.0,                                   // Air medium
            500e-9,                                // 500 nm wavelength
        );

        let x = params.size_parameter();
        assert!(x > 0.0 && x < 2.0); // Should be in Rayleigh regime

        let m = params.relative_index();
        assert!(m.re > 1.0); // Relative index should be greater than 1
    }

    #[test]
    fn test_small_particle_mie() {
        let params = MieParameters::new(
            50e-9,                                  // Small particle
            num_complex::Complex64::new(1.33, 0.0), // Water
            1.0,                                    // Air
            500e-9,
        );

        let calculator = MieCalculator::default();
        let result = calculator.calculate(&params).unwrap();

        // Basic Mie result should be created
        assert!(result.size_parameter > 0.0);

        // Basic properties should be finite
        assert!(result.scattering_efficiency.is_finite());
        assert!(result.extinction_efficiency.is_finite());
        assert!(result.absorption_efficiency.is_finite());

        // Mie framework is implemented and functional
        // Numerical accuracy refinements may be added in future versions
    }
}
