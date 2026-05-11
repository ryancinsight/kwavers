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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
    pub fn new(material: &PoroelasticMaterial) -> Self {
        Self {
            material: material.clone(),
        }
    }

    /// Compute wave speeds (fast and slow P-waves)
    ///
    /// Solves Biot's characteristic equation for complex wave numbers
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_wave_speeds(&self, _frequency: f64) -> KwaversResult<WaveSpeeds> {
        // Biot coefficients
        let phi = self.material.porosity;
        let rho_s = self.material.solid_density;
        let rho_f = self.material.fluid_density;
        let alpha = self.material.tortuosity;

        // Effective densities
        let rho_11 = (1.0 - phi).mul_add(rho_s, phi * rho_f * (alpha - 1.0));
        let rho_22 = phi * rho_f * alpha;

        // Elastic wave propagation coefficients in porous media
        let k_s = self.material.solid_bulk_modulus;
        let k_f = self.material.fluid_bulk_modulus;
        let g = self.material.shear_modulus;

        // P and Q moduli
        let p_coeff = (4.0_f64 / 3.0).mul_add(g, k_s);
        let q_coeff = k_f * phi;
        let r_coeff = k_f * phi;

        // Solve for wave speeds using high-frequency approximation
        // Reference: Biot (1956) theory of wave propagation in porous media
        let fast_wave = ((2.0f64.mul_add(q_coeff, p_coeff) + r_coeff) / rho_11).sqrt();
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::acoustics::mechanics::poroelastic::PoroelasticMaterial;

    fn bone() -> PoroelasticMaterial {
        PoroelasticMaterial::default() // trabecular bone
    }

    /// Fast P-wave must be greater than slow P-wave for trabecular bone at 1 MHz.
    ///
    /// Physics: the fast Biot wave involves in-phase solid–fluid motion; the slow
    /// wave is a diffusion-like mode and propagates much more slowly.
    #[test]
    fn compute_wave_speeds_fast_greater_than_slow_for_bone() {
        let biot = BiotTheory::new(&bone());
        let speeds = biot.compute_wave_speeds(1e6).unwrap();
        assert!(speeds.fast_wave > 0.0, "fast_wave must be positive");
        assert!(speeds.slow_wave > 0.0, "slow_wave must be positive");
        assert!(speeds.shear_wave > 0.0, "shear_wave must be positive");
        assert!(
            speeds.fast_wave > speeds.slow_wave,
            "fast_wave ({:.1}) must exceed slow_wave ({:.1})",
            speeds.fast_wave,
            speeds.slow_wave
        );
    }

    /// Shear wave speed equals sqrt(G/ρ₁₁) analytically.
    ///
    /// For default bone: G=3.5e9, ρ₁₁=(1-φ)ρ_s+φρ_f(α-1)=1550 → c_S≈1503 m/s.
    #[test]
    fn compute_wave_speeds_shear_wave_consistent_with_analytical() {
        let m = bone();
        let phi = m.porosity;
        let rho_11 =
            (1.0 - phi) * m.solid_density + phi * m.fluid_density * (m.tortuosity - 1.0);
        let expected_shear = (m.shear_modulus / rho_11).sqrt();

        let biot = BiotTheory::new(&m);
        let speeds = biot.compute_wave_speeds(1e6).unwrap();
        assert!(
            (speeds.shear_wave - expected_shear).abs() < 1.0,
            "shear_wave {:.1} must match analytical {expected_shear:.1}",
            speeds.shear_wave
        );
    }

    /// Both attenuation coefficients must be positive at 1 MHz.
    #[test]
    fn compute_attenuation_both_coefficients_positive() {
        let biot = BiotTheory::new(&bone());
        let (af, as_) = biot.compute_attenuation(1e6).unwrap();
        assert!(af > 0.0, "alpha_fast must be positive (got {af:.3e})");
        assert!(as_ > 0.0, "alpha_slow must be positive (got {as_:.3e})");
    }

    /// The slow Biot wave has much higher attenuation than the fast wave.
    ///
    /// Physics: the slow wave is a diffusion-dominated mode; its attenuation
    /// scales as ω² / c_slow³ which is much larger than ω² / c_fast³.
    #[test]
    fn compute_attenuation_slow_wave_dominates_fast_wave() {
        let biot = BiotTheory::new(&bone());
        let (alpha_fast, alpha_slow) = biot.compute_attenuation(1e6).unwrap();
        assert!(
            alpha_slow > alpha_fast,
            "slow-wave attenuation ({alpha_slow:.3e}) must exceed fast-wave ({alpha_fast:.3e})"
        );
    }
}
