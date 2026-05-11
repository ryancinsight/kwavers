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
            fluid_viscosity: 1e-3,      // Water at 20°C
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
                fluid_viscosity: 1e-3,
                tortuosity: 1.2,
            }),
            "liver" => Ok(Self {
                porosity: 0.15, // 15% vascular space
                solid_density: 1050.0,
                fluid_density: 1000.0,
                solid_bulk_modulus: 2.5e9, // 2.5 GPa
                fluid_bulk_modulus: 2.25e9,
                shear_modulus: 5e3, // 5 kPa (soft)
                permeability: 1e-11,
                fluid_viscosity: 1e-3,
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

    /// Calculate bulk density ρ = (1-φ)ρ_s + φρ_f
    #[must_use] 
    pub fn bulk_density(&self) -> f64 {
        (1.0 - self.porosity).mul_add(self.solid_density, self.porosity * self.fluid_density)
    }

    /// Calculate effective bulk modulus (Gassmann's equation)
    #[must_use] 
    pub fn effective_bulk_modulus(&self) -> f64 {
        let k_s = self.solid_bulk_modulus;
        let k_f = self.fluid_bulk_modulus;
        let phi = self.porosity;

        let term1 = (1.0 - phi) / k_s;
        let term2 = phi / k_f;
        1.0 / (term1 + term2)
    }

    /// Calculate characteristic frequency (Biot critical frequency)
    ///
    /// ω_c = (φ η) / (κ ρ_f α)
    #[must_use]
    pub fn characteristic_frequency(&self) -> f64 {
        let phi = self.porosity;
        let eta = self.fluid_viscosity;
        let kappa = self.permeability;
        let rho_f = self.fluid_density;
        let alpha = self.tortuosity;

        (phi * eta) / (kappa * rho_f * alpha)
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
        let expected = 1.0 / ((1.0 - m.porosity) / m.solid_bulk_modulus
            + m.porosity / m.fluid_bulk_modulus);
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
    /// Analytical for default bone:
    ///   φ=0.3, η=1e-3, κ=1e-9, ρ_f=1000, α=1.5.
    ///   ω_c = 0.3·1e-3 / (1e-9·1000·1.5) = 3e-4 / 1.5e-6 = 200 rad/s.
    #[test]
    fn characteristic_frequency_matches_analytical_biot_formula() {
        let m = PoroelasticMaterial::default();
        let expected =
            m.porosity * m.fluid_viscosity / (m.permeability * m.fluid_density * m.tortuosity);
        let got = m.characteristic_frequency();
        assert!(
            (got - expected).abs() < 1e-6,
            "characteristic_frequency: got {got:.3e} expected {expected:.3e}"
        );
        assert!((got - 200.0).abs() < 1e-6, "ω_c must be 200 rad/s for default bone");
    }

    /// `new` rejects porosity outside (0, 1].
    #[test]
    fn new_rejects_invalid_porosity() {
        let ok = || (0.3, 2000.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 1.5_f64);
        let (_, rs, rf, ks, kf, g, k, eta, tor) = ok();

        assert!(PoroelasticMaterial::new(-0.1, rs, rf, ks, kf, g, k, eta, tor).is_err(), "negative porosity");
        assert!(PoroelasticMaterial::new(1.5, rs, rf, ks, kf, g, k, eta, tor).is_err(), "porosity > 1");
    }

    /// `new` rejects non-positive densities.
    #[test]
    fn new_rejects_nonpositive_densities() {
        let (phi, _, rf, ks, kf, g, k, eta, tor) = (0.3, 2000.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 1.5_f64);
        assert!(PoroelasticMaterial::new(phi, 0.0, rf, ks, kf, g, k, eta, tor).is_err(), "zero solid_density");
        assert!(PoroelasticMaterial::new(phi, 2000.0, 0.0, ks, kf, g, k, eta, tor).is_err(), "zero fluid_density");
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
            assert!(m.bulk_density() > 0.0, "{tissue} bulk_density must be positive");
        }
    }

    /// `from_tissue_type` rejects an unknown tissue name.
    #[test]
    fn from_tissue_type_rejects_unknown_tissue() {
        assert!(PoroelasticMaterial::from_tissue_type("cartilage").is_err());
    }
}
