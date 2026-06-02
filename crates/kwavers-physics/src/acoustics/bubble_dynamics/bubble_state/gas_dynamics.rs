//! Gas types and species enumeration for composition specification

use kwavers_core::constants::thermodynamic::{
    HEAT_CAPACITY_RATIO_DIATOMIC, HEAT_CAPACITY_RATIO_MONATOMIC,
};

/// Gas type enumeration for composition specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GasType {
    N2,  // Nitrogen
    O2,  // Oxygen
    Ar,  // Argon
    He,  // Helium
    Xe,  // Xenon
    CO2, // Carbon dioxide
    H2O, // Water vapor (handled separately in most cases)
}

impl GasType {
    /// Get Van der Waals constant a [bar·L²/mol²]
    #[must_use]
    pub fn vdw_a(&self) -> f64 {
        match self {
            Self::N2 => 1.370, // From literature
            Self::O2 => 1.382,
            Self::Ar => 1.355,
            Self::He => 0.0346,
            Self::Xe => 4.250,
            Self::CO2 => 3.658,
            Self::H2O => 5.537,
        }
    }

    /// Get Van der Waals constant b [L/mol]
    #[must_use]
    pub fn vdw_b(&self) -> f64 {
        match self {
            Self::N2 => 0.0387, // From literature
            Self::O2 => 0.0319,
            Self::Ar => 0.0320,
            Self::He => 0.0238,
            Self::Xe => 0.0510,
            Self::CO2 => 0.0427,
            Self::H2O => 0.0305,
        }
    }

    /// Get molecular weight [kg/mol]
    #[must_use]
    pub fn molecular_weight(&self) -> f64 {
        match self {
            Self::N2 => 0.028014,
            Self::O2 => 0.031998,
            Self::Ar => 0.039948,
            Self::He => 0.004003,
            Self::Xe => 0.131293,
            Self::CO2 => 0.044009,
            Self::H2O => 0.018015,
        }
    }

    /// Get heat capacity ratio (gamma)
    #[must_use]
    pub fn gamma(&self) -> f64 {
        match self {
            Self::N2 => HEAT_CAPACITY_RATIO_DIATOMIC,  // Diatomic
            Self::O2 => HEAT_CAPACITY_RATIO_DIATOMIC,  // Diatomic
            Self::Ar => HEAT_CAPACITY_RATIO_MONATOMIC, // Monatomic
            Self::He => HEAT_CAPACITY_RATIO_MONATOMIC, // Monatomic
            Self::Xe => HEAT_CAPACITY_RATIO_MONATOMIC, // Monatomic
            Self::CO2 => 1.289,                        // Triatomic
            Self::H2O => 1.33,                         // Triatomic
        }
    }
}

/// Gas species in bubble
#[allow(clippy::module_inception)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GasSpecies {
    Air,
    Argon,
    Xenon,
    Nitrogen,
    Oxygen,
    Custom { gamma: f64, molecular_weight: f64 },
}

impl GasSpecies {
    /// Get polytropic index
    #[must_use]
    pub fn gamma(&self) -> f64 {
        match self {
            Self::Air => HEAT_CAPACITY_RATIO_DIATOMIC,
            Self::Argon => HEAT_CAPACITY_RATIO_MONATOMIC,
            Self::Xenon => HEAT_CAPACITY_RATIO_MONATOMIC,
            Self::Nitrogen => HEAT_CAPACITY_RATIO_DIATOMIC,
            Self::Oxygen => HEAT_CAPACITY_RATIO_DIATOMIC,
            Self::Custom { gamma, .. } => *gamma,
        }
    }

    /// Get molecular weight [kg/mol]
    #[must_use]
    pub fn molecular_weight(&self) -> f64 {
        match self {
            Self::Air => 0.029,
            Self::Argon => 0.040,
            Self::Xenon => 0.131,
            Self::Nitrogen => 0.028,
            Self::Oxygen => 0.032,
            Self::Custom {
                molecular_weight, ..
            } => *molecular_weight,
        }
    }

    /// Get molar heat capacity at constant volume [J/(mol·K)]
    ///
    /// These are fundamental thermodynamic properties of the gas species,
    /// not derived from gamma to maintain consistency with real gas models.
    #[must_use]
    pub fn molar_heat_capacity_cv(&self) -> f64 {
        use kwavers_core::constants::GAS_CONSTANT as R_GAS;

        match self {
            // For diatomic gases (Air, N2, O2): Cv = (5/2)R
            Self::Air => 2.5 * R_GAS,      // 20.8 J/(mol·K)
            Self::Nitrogen => 2.5 * R_GAS, // 20.8 J/(mol·K)
            Self::Oxygen => 2.5 * R_GAS,   // 20.8 J/(mol·K)

            // For monatomic gases (Ar, Xe): Cv = (3/2)R
            Self::Argon => 1.5 * R_GAS, // 12.5 J/(mol·K)
            Self::Xenon => 1.5 * R_GAS, // 12.5 J/(mol·K)

            // For custom gas, derive from gamma (with documented assumption)
            Self::Custom { gamma, .. } => R_GAS / (gamma - 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::GAS_CONSTANT;

    // ── GasType ──────────────────────────────────────────────────────────────

    /// Van der Waals constant `a` matches literature values (bar·L²/mol²).
    ///
    /// Reference: Atkins & de Paula, "Physical Chemistry" (10th ed.) Table 1C.3.
    #[test]
    fn gas_type_vdw_a_matches_literature() {
        assert!((GasType::N2.vdw_a() - 1.370).abs() < 1e-6, "N₂ a");
        assert!((GasType::He.vdw_a() - 0.0346).abs() < 1e-6, "He a");
        assert!((GasType::Xe.vdw_a() - 4.250).abs() < 1e-6, "Xe a");
    }

    /// Van der Waals constant `b` matches literature values (L/mol).
    #[test]
    fn gas_type_vdw_b_matches_literature() {
        assert!((GasType::N2.vdw_b() - 0.0387).abs() < 1e-6, "N₂ b");
        assert!((GasType::He.vdw_b() - 0.0238).abs() < 1e-6, "He b");
        assert!((GasType::O2.vdw_b() - 0.0319).abs() < 1e-6, "O₂ b");
    }

    /// Adiabatic index γ: monatomic = 5/3, diatomic = 1.4, triatomic = 1.289.
    ///
    /// Statistical mechanics: γ = 1 + 2/f where f is degrees of freedom.
    #[test]
    fn gas_type_gamma_by_molecular_structure() {
        assert!((GasType::N2.gamma() - 1.4).abs() < 1e-10, "N₂ diatomic");
        assert!((GasType::O2.gamma() - 1.4).abs() < 1e-10, "O₂ diatomic");
        assert!(
            (GasType::Ar.gamma() - 5.0 / 3.0).abs() < 1e-10,
            "Ar monatomic"
        );
        assert!(
            (GasType::He.gamma() - 5.0 / 3.0).abs() < 1e-10,
            "He monatomic"
        );
        assert!(
            (GasType::CO2.gamma() - 1.289).abs() < 1e-10,
            "CO₂ triatomic"
        );
    }

    // ── GasSpecies ───────────────────────────────────────────────────────────

    /// `GasSpecies::Air` is diatomic (γ=1.4); Argon/Xenon are monatomic (γ=5/3).
    #[test]
    fn gas_species_gamma_matches_species() {
        assert!((GasSpecies::Air.gamma() - 1.4).abs() < 1e-10, "Air γ=1.4");
        assert!(
            (GasSpecies::Argon.gamma() - 5.0 / 3.0).abs() < 1e-10,
            "Ar γ=5/3"
        );
        assert!(
            (GasSpecies::Xenon.gamma() - 5.0 / 3.0).abs() < 1e-10,
            "Xe γ=5/3"
        );
    }

    /// `Custom` variant stores gamma and molecular weight verbatim.
    #[test]
    fn gas_species_custom_stores_values_verbatim() {
        let g = 1.3_f64;
        let mw = 0.044_f64;
        let s = GasSpecies::Custom {
            gamma: g,
            molecular_weight: mw,
        };
        assert!((s.gamma() - g).abs() < 1e-15);
        assert!((s.molecular_weight() - mw).abs() < 1e-15);
    }

    /// Diatomic molar Cv = (5/2)R; monatomic Cv = (3/2)R.
    ///
    /// Statistical mechanics: each translational+rotational DOF contributes R/2.
    #[test]
    fn gas_species_molar_cv_matches_statistical_mechanics() {
        let cv_air = GasSpecies::Air.molar_heat_capacity_cv();
        let cv_ar = GasSpecies::Argon.molar_heat_capacity_cv();
        assert!((cv_air - 2.5 * GAS_CONSTANT).abs() < 1e-10, "Air Cv = 2.5R");
        assert!((cv_ar - 1.5 * GAS_CONSTANT).abs() < 1e-10, "Ar Cv = 1.5R");
    }
}
