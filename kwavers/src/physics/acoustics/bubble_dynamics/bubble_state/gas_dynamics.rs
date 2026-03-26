//! Gas types and species enumeration for composition specification

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
            Self::N2 => 1.4,       // Diatomic
            Self::O2 => 1.4,       // Diatomic
            Self::Ar => 5.0 / 3.0, // Monatomic
            Self::He => 5.0 / 3.0, // Monatomic
            Self::Xe => 5.0 / 3.0, // Monatomic
            Self::CO2 => 1.289,    // Triatomic
            Self::H2O => 1.33,     // Triatomic
        }
    }
}

/// Gas species in bubble
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
            Self::Air => 1.4,
            Self::Argon => 5.0 / 3.0,
            Self::Xenon => 5.0 / 3.0,
            Self::Nitrogen => 1.4,
            Self::Oxygen => 1.4,
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
        use crate::core::constants::GAS_CONSTANT as R_GAS;

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
