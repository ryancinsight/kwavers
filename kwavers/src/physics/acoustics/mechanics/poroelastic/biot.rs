//! Biot theory implementation
//!
//! Reference: Biot (1956) "Theory of propagation of elastic waves"

use crate::core::error::KwaversResult;
use crate::physics::acoustics::mechanics::poroelastic::{PoroelasticMaterial, WaveSpeeds};
use std::f64::consts::PI;

/// Biot theory for poroelastic wave propagation
#[derive(Debug)]
pub struct BiotTheory {
    material: PoroelasticMaterial,
}

impl BiotTheory {
    /// Create new Biot theory calculator
    pub fn new(material: &PoroelasticMaterial) -> Self {
        Self {
            material: material.clone(),
        }
    }

    /// Compute wave speeds (fast and slow P-waves)
    ///
    /// Solves Biot's characteristic equation for complex wave numbers
    pub fn compute_wave_speeds(&self, _frequency: f64) -> KwaversResult<WaveSpeeds> {
        // Biot coefficients
        let phi = self.material.porosity;
        let rho_s = self.material.solid_density;
        let rho_f = self.material.fluid_density;
        let alpha = self.material.tortuosity;

        // Effective densities
        let rho_11 = (1.0 - phi) * rho_s + phi * rho_f * (alpha - 1.0);
        let rho_22 = phi * rho_f * alpha;

        // Elastic wave propagation coefficients in porous media
        let k_s = self.material.solid_bulk_modulus;
        let k_f = self.material.fluid_bulk_modulus;
        let g = self.material.shear_modulus;

        // P and Q moduli
        let p_coeff = k_s + (4.0 / 3.0) * g;
        let q_coeff = k_f * phi;
        let r_coeff = k_f * phi;

        // Solve for wave speeds using high-frequency approximation
        // Reference: Biot (1956) theory of wave propagation in porous media
        let fast_wave = ((p_coeff + 2.0 * q_coeff + r_coeff) / rho_11).sqrt();
        let slow_wave = (r_coeff / rho_22).sqrt();

        let shear_wave = (g / rho_11).sqrt();

        Ok(WaveSpeeds {
            fast_wave,
            slow_wave,
            shear_wave,
        })
    }

    /// Compute attenuation coefficients for fast and slow waves
    ///
    /// Returns (alpha_fast, alpha_slow) in Np/m
    pub fn compute_attenuation(&self, frequency: f64) -> KwaversResult<(f64, f64)> {
        let omega = 2.0 * PI * frequency;
        let phi = self.material.porosity;
        let kappa = self.material.permeability;
        let eta = self.material.fluid_viscosity;

        // Viscous damping parameter
        let b = (phi * phi * eta) / kappa;

        // Approximate attenuation (high frequency limit)
        let speeds = self.compute_wave_speeds(frequency)?;

        // Fast wave: lower attenuation
        let alpha_fast =
            (b * omega * omega) / (2.0 * self.material.bulk_density() * speeds.fast_wave.powi(3));

        // Slow wave: much higher attenuation
        let alpha_slow =
            (b * omega * omega) / (2.0 * self.material.fluid_density * speeds.slow_wave.powi(3));

        Ok((alpha_fast, alpha_slow))
    }
}
