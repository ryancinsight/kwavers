//! Poroelastic Material Properties
//!
//! ## Physical Model
//!
//! A poroelastic material is a biphasic mixture of a solid elastic frame and
//! a saturating viscous fluid.  The material is fully characterised by nine
//! independent scalar parameters that appear in Biot's theory:
//!
//! | Symbol | Name | Unit | Constraint |
//! |--------|------|------|------------|
//! | φ | porosity | — | 0 < φ < 1 |
//! | ρ_s | solid density | kg/m³ | > 0 |
//! | ρ_f | fluid density | kg/m³ | > 0 |
//! | K_s | solid bulk modulus | Pa | > 0 |
//! | K_f | fluid bulk modulus | Pa | > 0 |
//! | G | shear modulus of drained frame | Pa | > 0 |
//! | κ | permeability | m² | > 0 |
//! | η | fluid dynamic viscosity | Pa·s | > 0 |
//! | α | tortuosity | — | ≥ 1 |
//!
//! ## Key Derived Quantities
//!
//! **Bulk density** (linear mixture rule):
//! ```text
//! ρ = (1 − φ) ρ_s + φ ρ_f
//! ```
//!
//! **Effective bulk modulus** (Gassmann's equation, Gassmann 1951):
//! ```text
//! K_eff = 1 / ((1 − φ)/K_s + φ/K_f)
//! ```
//! This is the harmonic mean weighted by phase volume fractions.  It reduces
//! to the Reuss lower bound for the undrained bulk modulus when frame stiffness
//! effects dominate, and matches the Wood (1955) acoustic formula for
//! soft-tissue limits.
//!
//! **Biot critical frequency** (Biot 1956 §6):
//! ```text
//! ω_c = φ η / (κ ρ_f α)
//! ```
//! Below ω_c the slow Biot wave is over-damped (diffusion regime).  Above ω_c
//! both fast and slow P-waves propagate with the high-frequency phase velocities
//! derived in `BiotTheory::compute_wave_speeds`.
//!
//! ## References
//!
//! - Biot M.A. (1956). J. Acoust. Soc. Am. 28(2), 168–191.
//! - Gassmann F. (1951). Vierteljahrschrift Naturf. Ges. Zürich 96, 1–23.
//! - Wood A.B. (1955). *A Textbook of Sound*. Bell & Hyman, London.

use crate::core::constants::cavitation::VISCOSITY_WATER;
use crate::core::constants::fundamental::DENSITY_TISSUE;
use crate::core::error::{KwaversError, KwaversResult};

/// Poroelastic material properties
///
/// Represents a biphasic material with solid matrix and fluid phase
#[derive(Debug, Clone)]
pub struct PoroelasticMaterial {
    /// Porosity (0 < φ < 1)
    pub porosity: f64,
    /// Solid density (kg/m³)
    pub solid_density: f64,
    /// Fluid density (kg/m³)
    pub fluid_density: f64,
    /// Solid bulk modulus (Pa)
    pub solid_bulk_modulus: f64,
    /// Fluid bulk modulus (Pa)
    pub fluid_bulk_modulus: f64,
    /// Shear modulus of drained frame (Pa)
    pub shear_modulus: f64,
    /// Permeability (m²)
    pub permeability: f64,
    /// Fluid viscosity (Pa·s)
    pub fluid_viscosity: f64,
    /// Tortuosity (α ≥ 1)
    pub tortuosity: f64,
}

impl Default for PoroelasticMaterial {
    fn default() -> Self {
        // Typical trabecular bone properties
        Self {
            porosity: 0.3,              // 30% porosity
            solid_density: 2000.0,      // kg/m³
            fluid_density: 1000.0,      // Water
            solid_bulk_modulus: 10e9,   // 10 GPa
            fluid_bulk_modulus: 2.25e9, // 2.25 GPa
            shear_modulus: 3.5e9,       // 3.5 GPa
            permeability: 1e-9,         // 1 nm² (Darcy)
            fluid_viscosity: VISCOSITY_WATER, // Water at 20°C — SSOT: cavitation::VISCOSITY_WATER
            tortuosity: 1.5,            // Typical for bone
        }
    }
}

impl PoroelasticMaterial {
    /// Create new poroelastic material with validation
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        porosity: f64,
        solid_density: f64,
        fluid_density: f64,
        solid_bulk_modulus: f64,
        fluid_bulk_modulus: f64,
        shear_modulus: f64,
        permeability: f64,
        fluid_viscosity: f64,
        tortuosity: f64,
    ) -> KwaversResult<Self> {
        if !(0.0..=1.0).contains(&porosity) {
            return Err(KwaversError::InvalidInput(
                "Porosity must be between 0 and 1".to_owned(),
            ));
        }
        if solid_density <= 0.0 || fluid_density <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Densities must be positive".to_owned(),
            ));
        }
        if solid_bulk_modulus <= 0.0 || fluid_bulk_modulus <= 0.0 || shear_modulus <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Moduli must be positive".to_owned(),
            ));
        }
        if permeability <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Permeability must be positive".to_owned(),
            ));
        }
        if tortuosity < 1.0 {
            return Err(KwaversError::InvalidInput(
                "Tortuosity must be ≥ 1".to_owned(),
            ));
        }

        Ok(Self {
            porosity,
            solid_density,
            fluid_density,
            solid_bulk_modulus,
            fluid_bulk_modulus,
            shear_modulus,
            permeability,
            fluid_viscosity,
            tortuosity,
        })
    }

    /// Create from tissue type
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn from_tissue_type(tissue: &str) -> KwaversResult<Self> {
        match tissue {
            "trabecular_bone" => Ok(Self::default()),
            "cortical_bone" => Ok(Self {
                porosity: 0.05, // 5% porosity
                solid_density: 2000.0,
                fluid_density: 1000.0,
                solid_bulk_modulus: 20e9, // 20 GPa
                fluid_bulk_modulus: 2.25e9,
                shear_modulus: 7e9,  // 7 GPa
                permeability: 1e-12, // Very low
                fluid_viscosity: VISCOSITY_WATER,
                tortuosity: 1.2,
            }),
            "liver" => Ok(Self {
                porosity: 0.15, // 15% vascular space
                solid_density: DENSITY_TISSUE, // SSOT: fundamental::DENSITY_TISSUE
                fluid_density: 1000.0,
                solid_bulk_modulus: 2.5e9, // 2.5 GPa
                fluid_bulk_modulus: 2.25e9,
                shear_modulus: 5e3, // 5 kPa (soft)
                permeability: 1e-11,
                fluid_viscosity: VISCOSITY_WATER,
                tortuosity: 1.3,
            }),
            "lung" => Ok(Self {
                porosity: 0.8,             // 80% air-filled
                solid_density: 300.0,      // Low density
                fluid_density: 1.2,        // Air
                solid_bulk_modulus: 1e6,   // Very soft
                fluid_bulk_modulus: 1.4e5, // Air at 1 atm
                shear_modulus: 1e3,        // 1 kPa
                permeability: 1e-8,        // High permeability
                fluid_viscosity: 1.8e-5,   // Air
                tortuosity: 2.0,           // Complex structure
            }),
            _ => Err(KwaversError::InvalidInput(format!(
                "Unknown tissue type: {}",
                tissue
            ))),
        }
    }

    /// Bulk density of the saturated mixture [kg/m³].
    ///
    /// ## Formula (linear volume-fraction mixture rule)
    ///
    /// ```text
    /// ρ = (1 − φ) ρ_s + φ ρ_f
    /// ```
    ///
    /// This is the exact result for a statistically homogeneous mixture where
    /// phase volumes are non-overlapping.  It is used as ρ₁₁ + ρ₁₂ in Biot's
    /// effective-density matrix when the coupling density ρ₁₂ = 0 (no tortuosity
    /// correction).  Here, `bulk_density` is the full ρ for momentum balance.
    #[must_use]
    pub fn bulk_density(&self) -> f64 {
        (1.0 - self.porosity).mul_add(self.solid_density, self.porosity * self.fluid_density)
    }

    /// Effective bulk modulus of the saturated porous medium [Pa].
    ///
    /// ## Theorem (Gassmann 1951 / Reuss–Wood lower bound)
    ///
    /// **Statement.** For a statically iso-stress mixture (stress is uniform
    /// across phases), the effective bulk modulus satisfies the harmonic mean:
    ///
    /// ```text
    /// 1/K_eff = (1 − φ)/K_s + φ/K_f
    /// ```
    ///
    /// **Proof sketch.** The total volumetric strain is the phase-averaged strain:
    /// ε = (1−φ)ε_s + φε_f.  Under uniform stress σ, ε_s = σ/K_s and ε_f = σ/K_f.
    /// Therefore K_eff = σ/ε = 1/[(1−φ)/K_s + φ/K_f].  (Wood 1955, §5.)
    ///
    /// **Bounds.** K_eff is the Reuss (lower) bound; the Voigt (upper) bound is
    /// K_Voigt = (1−φ)K_s + φK_f.  For poroelastic media K_f < K_eff < K_s.
    ///
    /// ## References
    ///
    /// - Gassmann F. (1951). Vierteljahrschrift Naturf. Ges. Zürich 96, 1–23.
    /// - Wood A.B. (1955). *A Textbook of Sound*. Bell & Hyman.
    /// - Mavko G., Mukerji T., Dvorkin J. (2009). *The Rock Physics Handbook*
    ///   §4.4. Cambridge University Press.
    #[must_use]
    pub fn effective_bulk_modulus(&self) -> f64 {
        let k_s = self.solid_bulk_modulus;
        let k_f = self.fluid_bulk_modulus;
        let phi = self.porosity;

        let term1 = (1.0 - phi) / k_s;
        let term2 = phi / k_f;
        1.0 / (term1 + term2)
    }

    /// Biot critical angular frequency [rad/s].
    ///
    /// ## Formula (Biot 1956 §6)
    ///
    /// ```text
    /// ω_c = φ η / (κ ρ_f α)
    /// ```
    ///
    /// ## Physical interpretation
    ///
    /// Below ω_c the slow Biot P-wave is in the viscosity-dominated (diffusion)
    /// regime and attenuates within a fraction of a wavelength.  Above ω_c both
    /// P-wave modes propagate with the phase velocities predicted by
    /// `BiotTheory::compute_wave_speeds` (high-frequency limit).
    ///
    /// The transition from diffusive to propagative is analogous to the skin
    /// depth in electromagnetism: δ = √(2η/(ω ρ_f φ κ)) where δ → 0 as ω → ∞.
    ///
    /// ## Reference
    ///
    /// Biot M.A. (1956). J. Acoust. Soc. Am. 28(2), 168–191, §6.
    #[must_use]
    pub fn characteristic_frequency(&self) -> f64 {
        (self.porosity * self.fluid_viscosity)
            / (self.permeability * self.fluid_density * self.tortuosity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `bulk_density` implements the linear mixture rule: ρ = (1-φ)ρ_s + φρ_f.
    ///
    /// Analytical for trabecular bone default:
    ///   φ=0.3, ρ_s=2000, ρ_f=1000 → ρ = 0.7·2000 + 0.3·1000 = 1700 kg/m³.
    #[test]
    fn bulk_density_follows_mixture_rule_for_default_bone() {
        let m = PoroelasticMaterial::default();
        let expected = (1.0 - m.porosity) * m.solid_density + m.porosity * m.fluid_density;
        assert!(
            (m.bulk_density() - expected).abs() < 1e-10,
            "bulk_density: got {}, expected {expected}",
            m.bulk_density()
        );
        // Numerical: 0.7·2000 + 0.3·1000 = 1700
        assert!((m.bulk_density() - 1700.0).abs() < 1e-6);
    }

    /// `effective_bulk_modulus` implements Gassmann's equation:
    /// K_eff = 1 / ((1-φ)/K_s + φ/K_f).
    ///
    /// Analytical for default bone:
    ///   φ=0.3, K_s=10e9, K_f=2.25e9.
    ///   K_eff = 1/(0.7/10e9 + 0.3/2.25e9) = 1/(7e-11 + 1.333e-10) ≈ 4.918e9 Pa.
    #[test]
    fn effective_bulk_modulus_follows_gassmann_equation() {
        let m = PoroelasticMaterial::default();
        let expected =
            1.0 / ((1.0 - m.porosity) / m.solid_bulk_modulus + m.porosity / m.fluid_bulk_modulus);
        let got = m.effective_bulk_modulus();
        assert!(
            (got - expected).abs() < 1.0, // 1 Pa tolerance
            "effective_bulk_modulus: got {got:.3e} expected {expected:.3e}"
        );
        // Sanity: between fluid and solid moduli
        assert!(got > m.fluid_bulk_modulus * 0.5, "K_eff must exceed K_f/2");
        assert!(got < m.solid_bulk_modulus, "K_eff must be less than K_s");
    }

    /// `characteristic_frequency` implements ω_c = φ·η / (κ·ρ_f·α).
    ///
    /// Analytical for default trabecular bone:
    ///   φ=0.3, η=VISCOSITY_WATER=1.002e-3 Pa·s, κ=1e-9 m², ρ_f=1000 kg/m³, α=1.5.
    ///   ω_c = 0.3·1.002e-3 / (1e-9·1000·1.5) = 3.006e-4 / 1.5e-6 = 200.4 rad/s.
    #[test]
    fn characteristic_frequency_matches_analytical_biot_formula() {
        let m = PoroelasticMaterial::default();
        // Reference value derived from SSOT VISCOSITY_WATER (1.002e-3 Pa·s):
        let expected_analytical = 0.3 * VISCOSITY_WATER / (1e-9 * 1000.0 * 1.5);
        let expected_from_struct =
            m.porosity * m.fluid_viscosity / (m.permeability * m.fluid_density * m.tortuosity);
        let got = m.characteristic_frequency();
        // Struct-derived and analytical must agree.
        assert!(
            (got - expected_from_struct).abs() < 1e-6,
            "characteristic_frequency: got {got:.3e} expected {expected_from_struct:.3e}"
        );
        // Analytical value from SSOT constants.
        assert!(
            (got - expected_analytical).abs() < 1e-6,
            "ω_c must be {expected_analytical:.3} rad/s for default bone, got {got:.3}"
        );
    }

    /// `new` rejects porosity outside (0, 1].
    #[test]
    fn new_rejects_invalid_porosity() {
        let ok = || {
            (
                0.3, 2000.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 1.5_f64,
            )
        };
        let (_, rs, rf, ks, kf, g, k, eta, tor) = ok();

        assert!(
            PoroelasticMaterial::new(-0.1, rs, rf, ks, kf, g, k, eta, tor).is_err(),
            "negative porosity"
        );
        assert!(
            PoroelasticMaterial::new(1.5, rs, rf, ks, kf, g, k, eta, tor).is_err(),
            "porosity > 1"
        );
    }

    /// `new` rejects non-positive densities.
    #[test]
    fn new_rejects_nonpositive_densities() {
        let (phi, _, rf, ks, kf, g, k, eta, tor) = (
            0.3, 2000.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 1.5_f64,
        );
        assert!(
            PoroelasticMaterial::new(phi, 0.0, rf, ks, kf, g, k, eta, tor).is_err(),
            "zero solid_density"
        );
        assert!(
            PoroelasticMaterial::new(phi, 2000.0, 0.0, ks, kf, g, k, eta, tor).is_err(),
            "zero fluid_density"
        );
    }

    /// `new` rejects tortuosity < 1.
    #[test]
    fn new_rejects_tortuosity_below_one() {
        let r = PoroelasticMaterial::new(0.3, 2000.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 0.8);
        assert!(r.is_err(), "tortuosity=0.8 < 1.0 must be rejected");
    }

    /// `from_tissue_type` accepts all four defined tissue strings.
    #[test]
    fn from_tissue_type_accepts_all_known_tissues() {
        for tissue in &["trabecular_bone", "cortical_bone", "liver", "lung"] {
            let r = PoroelasticMaterial::from_tissue_type(tissue);
            assert!(r.is_ok(), "{tissue} must be a known tissue type");
            let m = r.unwrap();
            assert!(
                m.bulk_density() > 0.0,
                "{tissue} bulk_density must be positive"
            );
        }
    }

    /// `from_tissue_type` rejects an unknown tissue name.
    #[test]
    fn from_tissue_type_rejects_unknown_tissue() {
        assert!(PoroelasticMaterial::from_tissue_type("cartilage").is_err());
    }
}
