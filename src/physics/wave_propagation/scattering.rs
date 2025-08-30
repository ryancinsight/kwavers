//! Unified scattering module for wave propagation
//!
//! This module consolidates all scattering phenomena including:
//! - Particle scattering (Rayleigh, Mie)
//! - Surface scattering (rough interfaces)
//! - Volume scattering (inhomogeneous media)
//!
//! # Literature References
//!
//! 1. **Bohren, C. F., & Huffman, D. R. (1983)**. "Absorption and Scattering
//!    of Light by Small Particles." Wiley. ISBN: 978-0471293408
//!    - Comprehensive Mie theory
//!    - Rayleigh approximation limits
//!
//! 2. **Ishimaru, A. (1978)**. "Wave Propagation and Scattering in Random Media."
//!    Academic Press. ISBN: 978-0123747013
//!    - Multiple scattering theory
//!    - Transport equation derivation
//!
//! 3. **van de Hulst, H. C. (1981)**. "Light Scattering by Small Particles."
//!    Dover Publications. ISBN: 978-0486642284
//!    - Classical scattering theory

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;
use std::f64::consts::PI;

// Scattering regime thresholds
/// Rayleigh regime upper bound (ka < 0.1)
const RAYLEIGH_REGIME_THRESHOLD: f64 = 0.1;
/// Mie regime upper bound (0.1 < ka < 10.0)
const MIE_REGIME_THRESHOLD: f64 = 10.0;

// Scattering coefficients
/// Rayleigh scattering coefficient (8π/3)
const RAYLEIGH_COEFFICIENT: f64 = 8.0 * PI / 3.0;
/// Rayleigh-Gans angular factor coefficient
const RAYLEIGH_GANS_ANGULAR_COEFFICIENT: f64 = 3.0 / (16.0 * PI);
/// Isotropic phase function normalization
const ISOTROPIC_PHASE_NORMALIZATION: f64 = 1.0 / (4.0 * PI);

/// Scattering regime based on size parameter
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScatteringRegime {
    /// Rayleigh scattering (ka << 1)
    Rayleigh,
    /// Mie scattering (ka ≈ 1)
    Mie,
    /// Geometric scattering (ka >> 1)
    Geometric,
}

/// Unified scattering calculator
#[derive(Debug, Debug)]
pub struct ScatteringCalculator {
    /// Wave frequency [Hz]
    frequency: f64,
    /// Wavelength [m]
    wavelength: f64,
    /// Wave number k = 2π/λ
    wave_number: f64,
}

impl ScatteringCalculator {
    /// Create a new scattering calculator
    pub fn new(frequency: f64, wave_speed: f64) -> Self {
        let wavelength = wave_speed / frequency;
        let wave_number = 2.0 * PI / wavelength;

        Self {
            frequency,
            wavelength,
            wave_number,
        }
    }

    /// Determine scattering regime based on size parameter
    pub fn determine_regime(&self, particle_radius: f64) -> ScatteringRegime {
        let size_parameter = self.wave_number * particle_radius;

        if size_parameter < RAYLEIGH_REGIME_THRESHOLD {
            ScatteringRegime::Rayleigh
        } else if size_parameter < MIE_REGIME_THRESHOLD {
            ScatteringRegime::Mie
        } else {
            ScatteringRegime::Geometric
        }
    }

    /// Calculate Rayleigh scattering cross-section
    /// σ_s = (8π/3) * k^4 * a^6 * |m^2 - 1 / m^2 + 2|^2
    pub fn rayleigh_cross_section(&self, particle_radius: f64, refractive_index_ratio: f64) -> f64 {
        let ka = self.wave_number * particle_radius;
        let ka2 = ka * ka;
        let ka4 = ka2 * ka2;
        let a6 = particle_radius.powi(6);

        let m2 = refractive_index_ratio * refractive_index_ratio;
        let polarizability = (m2 - 1.0) / (m2 + 2.0);

        RAYLEIGH_COEFFICIENT * ka4 * a6 * polarizability * polarizability
    }

    /// Calculate Rayleigh scattering amplitude
    pub fn rayleigh_amplitude(
        &self,
        particle_radius: f64,
        refractive_index_ratio: f64,
        scattering_angle: f64,
    ) -> f64 {
        let ka = self.wave_number * particle_radius;
        let ka3 = ka * ka * ka;

        let m2 = refractive_index_ratio * refractive_index_ratio;
        let polarizability = (m2 - 1.0) / (m2 + 2.0);

        // Angular dependence: 1 + cos²(θ) for unpolarized light
        let angular_factor = 1.0 + scattering_angle.cos().powi(2);

        ka3 * polarizability * angular_factor.sqrt()
    }

    /// Calculate Mie scattering coefficients (simplified)
    /// Full Mie theory requires complex Bessel functions
    pub fn mie_coefficients(
        &self,
        particle_radius: f64,
        refractive_index_ratio: f64,
        max_order: usize,
    ) -> KwaversResult<(Vec<f64>, Vec<f64>)> {
        let size_parameter = self.wave_number * particle_radius;

        // Simplified Mie coefficients for real refractive indices
        let mut a_n = Vec::with_capacity(max_order);
        let mut b_n = Vec::with_capacity(max_order);

        for n in 1..=max_order {
            let n_f = n as f64;

            // Simplified coefficients (exact calculation requires Bessel functions)
            // Using asymptotic approximations for demonstration
            let denominator = n_f * n_f + size_parameter * size_parameter;

            let a = 2.0 * n_f * (refractive_index_ratio - 1.0) / denominator;
            let b = 2.0 * n_f * (1.0 / refractive_index_ratio - 1.0) / denominator;

            a_n.push(a);
            b_n.push(b);
        }

        Ok((a_n, b_n))
    }

    /// Calculate total scattering cross-section
    pub fn total_cross_section(
        &self,
        particle_radius: f64,
        refractive_index_ratio: f64,
    ) -> KwaversResult<f64> {
        let regime = self.determine_regime(particle_radius);

        match regime {
            ScatteringRegime::Rayleigh => {
                Ok(self.rayleigh_cross_section(particle_radius, refractive_index_ratio))
            }
            ScatteringRegime::Mie => {
                // Simplified Mie cross-section
                let (a_n, b_n) =
                    self.mie_coefficients(particle_radius, refractive_index_ratio, 10)?;

                let mut cross_section = 0.0;
                for (n, (a, b)) in a_n.iter().zip(b_n.iter()).enumerate() {
                    let n_f = (n + 1) as f64;
                    cross_section += (2.0 * n_f + 1.0) * (a * a + b * b);
                }

                Ok(2.0 * PI / (self.wave_number * self.wave_number) * cross_section)
            }
            ScatteringRegime::Geometric => {
                // Geometric optics limit
                Ok(PI * particle_radius * particle_radius)
            }
        }
    }

    /// Calculate differential scattering cross-section
    pub fn differential_cross_section(
        &self,
        particle_radius: f64,
        refractive_index_ratio: f64,
        scattering_angle: f64,
    ) -> KwaversResult<f64> {
        let regime = self.determine_regime(particle_radius);

        match regime {
            ScatteringRegime::Rayleigh => {
                let total = self.rayleigh_cross_section(particle_radius, refractive_index_ratio);
                // Rayleigh angular distribution: (1 + cos²θ) / 2
                let angular = (1.0 + scattering_angle.cos().powi(2)) / 2.0;
                Ok(total * angular / (4.0 * PI))
            }
            _ => {
                // Simplified for Mie and geometric
                let total = self.total_cross_section(particle_radius, refractive_index_ratio)?;
                Ok(total / (4.0 * PI))
            }
        }
    }
}

/// Volume scattering in inhomogeneous media
#[derive(Debug, Debug)]
pub struct VolumeScattering {
    /// Scattering coefficient [1/m]
    pub scattering_coefficient: Array3<f64>,
    /// Anisotropy factor g = <cos θ>
    pub anisotropy: Array3<f64>,
    /// Phase function type
    pub phase_function: PhaseFunction,
}

/// Scattering phase function
#[derive(Debug, Clone, Copy)]
pub enum PhaseFunction {
    /// Isotropic scattering
    Isotropic,
    /// Henyey-Greenstein phase function
    HenyeyGreenstein,
    /// Rayleigh phase function
    Rayleigh,
    /// Mie phase function (tabulated)
    Mie,
}

impl VolumeScattering {
    /// Create volume scattering from medium properties
    pub fn from_medium(grid: &Grid, medium: &dyn Medium) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        let mut scattering_coefficient = Array3::zeros(shape);
        let mut anisotropy = Array3::zeros(shape);

        // Sample medium properties at grid points
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    // Get scattering properties from medium
                    // Using optical scattering coefficient as example
                    scattering_coefficient[(i, j, k)] =
                        medium.optical_scattering_coefficient(x, y, z, grid);

                    // Default anisotropy for tissue
                    anisotropy[(i, j, k)] = 0.9; // Forward scattering
                }
            }
        }

        Self {
            scattering_coefficient,
            anisotropy,
            phase_function: PhaseFunction::HenyeyGreenstein,
        }
    }

    /// Evaluate phase function
    pub fn phase_function_value(&self, cos_theta: f64, g: f64) -> f64 {
        match self.phase_function {
            PhaseFunction::Isotropic => 1.0 / (4.0 * PI),
            PhaseFunction::HenyeyGreenstein => {
                // p(cos θ) = (1 - g²) / (4π * (1 + g² - 2g*cos θ)^(3/2))
                let numerator = 1.0 - g * g;
                let denominator = 4.0 * PI * (1.0 + g * g - 2.0 * g * cos_theta).powf(1.5);
                numerator / denominator
            }
            PhaseFunction::Rayleigh => {
                // p(cos θ) = (3/16π) * (1 + cos² θ)
                (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta)
            }
            PhaseFunction::Mie => {
                // Simplified - would need tabulated values
                self.phase_function_value(cos_theta, g) // Use HG as approximation
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scattering_regime() {
        let calc = ScatteringCalculator::new(1e6, 1500.0); // 1 MHz in water

        // Small particle - Rayleigh
        assert_eq!(calc.determine_regime(1e-6), ScatteringRegime::Rayleigh);

        // Medium particle - Mie
        assert_eq!(calc.determine_regime(1e-3), ScatteringRegime::Mie);

        // Large particle - Geometric
        assert_eq!(calc.determine_regime(1e-1), ScatteringRegime::Geometric);
    }

    #[test]
    fn test_rayleigh_cross_section() {
        let calc = ScatteringCalculator::new(5e14, 3e8); // Green light

        let radius = 50e-9; // 50 nm particle
        let n_ratio = 1.5; // Glass in air

        let cross_section = calc.rayleigh_cross_section(radius, n_ratio);

        // Cross section should be positive and small
        assert!(cross_section > 0.0);
        assert!(cross_section < PI * radius * radius); // Less than geometric
    }

    #[test]
    fn test_phase_function_normalization() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let vol = VolumeScattering {
            scattering_coefficient: Array3::from_elem((10, 10, 10), 1.0),
            anisotropy: Array3::from_elem((10, 10, 10), 1.0) * 0.9,
            phase_function: PhaseFunction::HenyeyGreenstein,
        };

        // Test for various anisotropy values
        for g in [0.0, 0.5, 0.9] {
            // Integrate phase function over solid angle (should equal 1)
            let n_samples = 1000; // Increase samples for better accuracy
            let mut integral = 0.0;

            for i in 0..n_samples {
                let cos_theta = -1.0 + 2.0 * (i as f64 + 0.5) / (n_samples as f64);
                let d_cos_theta = 2.0 / (n_samples as f64);

                let p = vol.phase_function_value(cos_theta, g);
                integral += p * 2.0 * PI * d_cos_theta;
            }

            // Each should integrate to approximately 1
            assert!(
                (integral - 1.0).abs() < 0.01,
                "Phase function normalization failed for g={}: integral={}",
                g,
                integral
            );
        }
    }
}
