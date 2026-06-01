//! Biot Theory of Poroelastic Wave Propagation
//!
//! ## Mathematical Foundation
//!
//! Biot (1956) derived the governing equations for wave propagation in a
//! fluid-saturated porous medium by coupling the elastic solid frame to the
//! viscous pore fluid through Darcy's law.  In the frequency domain the
//! coupled momentum equations are:
//!
//! ```text
//! −ω² (ρ₁₁ u + ρ₁₂ w) = ∇ · σ_s
//! −ω² (ρ₁₂ u + ρ₂₂ w) = ∇ · σ_f − iω (b/φ) w
//! ```
//!
//! where u and w are the solid and fluid displacement vectors, σ_s and σ_f
//! are the partial stress tensors, and b = φ²η/κ is the Darcy drag coefficient.
//!
//! ## Theorem (Two Compressional Waves)
//!
//! **Statement.** For a homogeneous isotropic poroelastic medium, the
//! dispersion relation factors into two longitudinal (P-wave) modes and one
//! shear (S-wave) mode.  The two P-wave wavenumbers k±² satisfy the
//! quadratic characteristic equation:
//!
//! ```text
//! (PR − Q²) k⁴ − ω²(ρ₁₁ R − 2ρ₁₂ Q + ρ₂₂ P) k² + ω⁴ (ρ₁₁ρ₂₂ − ρ₁₂²) = 0
//! ```
//!
//! where P = (4G/3 + K_s), Q = K_f φ, R = K_f φ are the Biot elastic
//! coefficients.  The fast P-wave (k₊) involves nearly in-phase solid–fluid
//! motion; the slow P-wave (k₋) is a diffusion-dominated mode with imaginary
//! wavenumber at sub-critical frequencies.
//!
//! **Proof sketch.** Apply plane-wave ansatz u, w ~ exp(ik·x − iωt) to the
//! coupled momentum equations, reduce to a 4×4 matrix eigenvalue problem, and
//! use the algebraic identity det(block-2×2) to factor into the two scalar
//! quadratics above.  (Biot 1956, §5; Pride & Berryman 2003, Appendix A.)
//!
//! ## High-Frequency Limit
//!
//! Above the Biot critical frequency ω_c = φη/(κρ_f α), viscous coupling
//! becomes negligible and the wave speeds approach the undrained (Gassmann)
//! limits.  The implementation uses these high-frequency expressions:
//!
//! ```text
//! c_fast  ≈ √((P + 2Q + R) / ρ₁₁)
//! c_slow  ≈ √(R / ρ₂₂)
//! c_shear ≈ √(G / ρ₁₁)
//! ```
//!
//! At low frequencies the slow wave attenuates exponentially with distance;
//! its spatial attenuation coefficient scales as α_slow ∝ ω²/c_slow³ which
//! exceeds the fast-wave coefficient by several orders of magnitude.
//!
//! ## References
//!
//! - Biot M.A. (1956). "Theory of propagation of elastic waves in a
//!   fluid-saturated porous solid." J. Acoust. Soc. Am. 28(2), 168–191.
//!   DOI: 10.1121/1.1908239
//! - Biot M.A. (1956). "Mechanics of deformation and acoustic propagation
//!   in porous media." J. Appl. Phys. 33(4), 1482–1498.
//! - Pride S.R., Berryman J.G. (2003). "Linear dynamics of double-porosity
//!   dual-permeability materials." Phys. Rev. E 68, 036604.
//! - Gassmann F. (1951). "Elasticity of porous media." Vierteljahrschrift
//!   der Naturforschenden Gesellschaft in Zürich 96, 1–23.

use crate::core::constants::numerical::TWO_PI;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::mechanics::poroelastic::{PoroelasticMaterial, WaveSpeeds};

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

    /// Compute wave speeds for fast P-wave, slow P-wave, and shear wave.
    ///
    /// ## Algorithm (high-frequency limit, ω >> ω_c)
    ///
    /// **Step 1.** Compute the Biot effective density tensors:
    ///
    /// ```text
    /// ρ₁₁ = (1 − φ) ρ_s + φ ρ_f (α − 1)   [solid effective inertia]
    /// ρ₂₂ = φ ρ_f α                          [fluid effective inertia]
    /// ```
    ///
    /// **Step 2.** Compute Biot elastic coefficients (Biot 1956 §3):
    ///
    /// ```text
    /// P = K_s + 4G/3    [undrained P-wave modulus of frame]
    /// Q = K_f φ         [coupling coefficient]
    /// R = K_f φ         [fluid modulus contribution]
    /// ```
    ///
    /// **Step 3.** High-frequency wave speed expressions (viscous drag → 0):
    ///
    /// ```text
    /// c_fast  = √((P + 2Q + R) / ρ₁₁)   [in-phase solid–fluid mode]
    /// c_slow  = √(R / ρ₂₂)               [out-of-phase diffusion mode]
    /// c_shear = √(G / ρ₁₁)               [pure solid shear mode]
    /// ```
    ///
    /// ## Theorem (fast > slow)
    ///
    /// Since P + 2Q + R > R and ρ₁₁ ≥ ρ₂₂ is not guaranteed, we appeal to the
    /// physical argument: the fast wave transmits stress through the stiffest
    /// (frame + fluid) combination, while the slow wave propagates only through
    /// the fluid bulk modulus with the fluid inertia.  Numerically:
    /// `c_fast / c_slow = √((P + 2Q + R) · ρ₂₂ / (R · ρ₁₁))`.
    /// For typical bone parameters (P ≈ 10 GPa, R ≈ 0.675 GPa), the ratio is
    /// approximately 3–5, confirming c_fast >> c_slow.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub fn compute_wave_speeds(&self, _frequency: f64) -> KwaversResult<WaveSpeeds> {
        let phi = self.material.porosity;
        let rho_s = self.material.solid_density;
        let rho_f = self.material.fluid_density;
        let alpha = self.material.tortuosity;

        // Step 1: effective density components (Biot 1956 eq. 2.1 and 2.2)
        // ρ₁₁ = (1−φ)ρ_s + φρ_f(α−1): solid frame inertia including induced fluid mass
        // ρ₂₂ = φρ_f·α: fluid inertia including tortuosity correction
        let rho_11 = (1.0 - phi).mul_add(rho_s, phi * rho_f * (alpha - 1.0));
        let rho_22 = phi * rho_f * alpha;

        // Step 2: Biot elastic coefficients (Biot 1956 §3)
        let k_s = self.material.solid_bulk_modulus;
        let k_f = self.material.fluid_bulk_modulus;
        let g = self.material.shear_modulus;

        // P = K_s + 4G/3 (undrained uniaxial modulus of drained frame)
        let p_coeff = (4.0_f64 / 3.0).mul_add(g, k_s);
        // Q = R = K_f·φ (fluid–frame coupling and fluid modulus)
        let q_coeff = k_f * phi;
        let r_coeff = k_f * phi;

        // Step 3: high-frequency wave speeds (Biot 1956, eq. 5.7–5.9)
        let fast_wave = ((2.0f64.mul_add(q_coeff, p_coeff) + r_coeff) / rho_11).sqrt();
        let slow_wave = (r_coeff / rho_22).sqrt();
        let shear_wave = (g / rho_11).sqrt();

        Ok(WaveSpeeds {
            fast_wave,
            slow_wave,
            shear_wave,
        })
    }

    /// Compute spatial attenuation coefficients for fast and slow P-waves.
    ///
    /// ## Formula (high-frequency limit, Biot 1956 §6)
    ///
    /// The viscous damping parameter b = φ²η/κ [Pa·s/m²] drives attenuation.
    /// In the high-frequency regime the imaginary part of the wavenumber gives:
    ///
    /// ```text
    /// α_fast = b·ω² / (2·ρ_bulk·c_fast³)   [Np/m]
    /// α_slow = b·ω² / (2·ρ_f  ·c_slow³)   [Np/m]
    /// ```
    ///
    /// ## Theorem (slow wave dominates)
    ///
    /// Since c_slow << c_fast and ρ_f < ρ_bulk, we have:
    ///
    /// ```text
    /// α_slow / α_fast  ≈  (ρ_bulk / ρ_f) · (c_fast / c_slow)³  >> 1
    /// ```
    ///
    /// The slow Biot wave is therefore orders of magnitude more attenuated than
    /// the fast wave.  For trabecular bone at 1 MHz (c_fast≈3000, c_slow≈700 m/s),
    /// the ratio α_slow/α_fast ≈ 1.7 · (3000/700)³ ≈ 134.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn compute_attenuation(&self, frequency: f64) -> KwaversResult<(f64, f64)> {
        let omega = TWO_PI * frequency;
        let phi = self.material.porosity;
        let kappa = self.material.permeability;
        let eta = self.material.fluid_viscosity;

        // Darcy drag coefficient b = φ²η/κ (Biot 1956 eq. 6.7)
        let b = (phi * phi * eta) / kappa;

        let speeds = self.compute_wave_speeds(frequency)?;

        // High-frequency attenuation: α = b·ω² / (2·ρ·c³)
        let alpha_fast =
            (b * omega * omega) / (2.0 * self.material.bulk_density() * speeds.fast_wave.powi(3));
        let alpha_slow =
            (b * omega * omega) / (2.0 * self.material.fluid_density * speeds.slow_wave.powi(3));

        Ok((alpha_fast, alpha_slow))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::numerical::MHZ_TO_HZ;
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
        let speeds = biot.compute_wave_speeds(MHZ_TO_HZ).unwrap();
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
        let rho_11 = (1.0 - phi) * m.solid_density + phi * m.fluid_density * (m.tortuosity - 1.0);
        let expected_shear = (m.shear_modulus / rho_11).sqrt();

        let biot = BiotTheory::new(&m);
        let speeds = biot.compute_wave_speeds(MHZ_TO_HZ).unwrap();
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
        let (af, as_) = biot.compute_attenuation(MHZ_TO_HZ).unwrap();
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
        let (alpha_fast, alpha_slow) = biot.compute_attenuation(MHZ_TO_HZ).unwrap();
        assert!(
            alpha_slow > alpha_fast,
            "slow-wave attenuation ({alpha_slow:.3e}) must exceed fast-wave ({alpha_fast:.3e})"
        );
    }
}
