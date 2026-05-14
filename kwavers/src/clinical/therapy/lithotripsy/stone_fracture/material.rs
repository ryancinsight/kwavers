//! Stone material properties composing canonical domain types.

use crate::domain::medium::properties::{ElasticPropertyData, StrengthPropertyData};

/// Stone material properties composing canonical domain types.
///
/// # References
///
/// - Williams et al. (2003): "Mechanical properties of urinary stones"
/// - Zohdi & Kuypers (2006): "Modeling and simulation of kidney stone fragmentation"
#[derive(Debug, Clone)]
pub struct StoneMaterial {
    elastic: ElasticPropertyData,
    strength: StrengthPropertyData,
}

impl StoneMaterial {
    /// Construct stone material from canonical domain property types.
    #[must_use]
    pub fn new(elastic: ElasticPropertyData, strength: StrengthPropertyData) -> Self {
        Self { elastic, strength }
    }

    /// Get elastic properties.
    #[inline]
    #[must_use]
    pub fn elastic(&self) -> &ElasticPropertyData {
        &self.elastic
    }

    /// Get strength properties.
    #[inline]
    #[must_use]
    pub fn strength(&self) -> &StrengthPropertyData {
        &self.strength
    }

    /// Density (kg/m³).
    #[inline]
    #[must_use]
    pub fn density(&self) -> f64 {
        self.elastic.density
    }

    /// Young's modulus (Pa).
    #[inline]
    #[must_use]
    pub fn youngs_modulus(&self) -> f64 {
        self.elastic.youngs_modulus()
    }

    /// Poisson's ratio (dimensionless).
    #[inline]
    #[must_use]
    pub fn poisson_ratio(&self) -> f64 {
        self.elastic.poisson_ratio()
    }

    /// Tensile strength (Pa) — uses ultimate tensile strength as fracture threshold.
    #[inline]
    #[must_use]
    pub fn tensile_strength(&self) -> f64 {
        self.strength.ultimate_strength
    }

    /// Calcium oxalate monohydrate (most common kidney stone).
    ///
    /// - Density: 2000 kg/m³, E: 20 GPa, ν: 0.3, σ_uts: 5 MPa
    /// # Panics
    /// - Panics if `Valid stone strength parameters`.
    ///
    #[must_use]
    pub fn calcium_oxalate_monohydrate() -> Self {
        let density = 2000.0;
        let youngs_modulus = 20e9;
        let poisson_ratio = 0.3;
        let tensile_strength = 5e6;
        let yield_strength = 0.8 * tensile_strength;
        let hardness = 3.0 * yield_strength;
        let fatigue_exponent = 15.0;

        let elastic = ElasticPropertyData::from_engineering(density, youngs_modulus, poisson_ratio);
        let strength =
            StrengthPropertyData::new(yield_strength, tensile_strength, hardness, fatigue_exponent)
                .expect("Valid stone strength parameters");

        Self { elastic, strength }
    }

    /// Uric acid stone (softer, more deformable).
    ///
    /// - Density: 1800 kg/m³, E: 10 GPa, ν: 0.35, σ_uts: 3 MPa
    /// # Panics
    /// - Panics if `Valid stone strength parameters`.
    ///
    #[must_use]
    pub fn uric_acid() -> Self {
        let density = 1800.0;
        let youngs_modulus = 10e9;
        let poisson_ratio = 0.35;
        let tensile_strength = 3e6;
        let yield_strength = 0.8 * tensile_strength;
        let hardness = 3.0 * yield_strength;
        let fatigue_exponent = 12.0;

        let elastic = ElasticPropertyData::from_engineering(density, youngs_modulus, poisson_ratio);
        let strength =
            StrengthPropertyData::new(yield_strength, tensile_strength, hardness, fatigue_exponent)
                .expect("Valid stone strength parameters");

        Self { elastic, strength }
    }

    /// Cystine stone (hardest, most resistant).
    ///
    /// - Density: 2100 kg/m³, E: 30 GPa, ν: 0.28, σ_uts: 8 MPa
    /// # Panics
    /// - Panics if `Valid stone strength parameters`.
    ///
    #[must_use]
    pub fn cystine() -> Self {
        let density = 2100.0;
        let youngs_modulus = 30e9;
        let poisson_ratio = 0.28;
        let tensile_strength = 8e6;
        let yield_strength = 0.8 * tensile_strength;
        let hardness = 3.0 * yield_strength;
        let fatigue_exponent = 18.0;

        let elastic = ElasticPropertyData::from_engineering(density, youngs_modulus, poisson_ratio);
        let strength =
            StrengthPropertyData::new(yield_strength, tensile_strength, hardness, fatigue_exponent)
                .expect("Valid stone strength parameters");

        Self { elastic, strength }
    }
}
