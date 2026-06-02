//! Material properties and damage parameters for cavitation modeling

// Material constants for stainless steel 316
const STAINLESS_STEEL_316_YIELD_STRENGTH: f64 = 290e6; // Pa
const STAINLESS_STEEL_316_ULTIMATE_STRENGTH: f64 = 580e6; // Pa
const STAINLESS_STEEL_316_HARDNESS: f64 = 2.0e9; // Pa
const STAINLESS_STEEL_316_DENSITY: f64 = 7850.0; // kg/m³
const DEFAULT_FATIGUE_EXPONENT: f64 = 3.0;
const DEFAULT_EROSION_RESISTANCE: f64 = 1.0;

use kwavers_core::constants::cavitation::{
    DEFAULT_CONCENTRATION_FACTOR, DEFAULT_FATIGUE_RATE, DEFAULT_PIT_EFFICIENCY,
    DEFAULT_THRESHOLD_PRESSURE,
};

/// Material properties for damage calculation
#[derive(Debug, Clone)]
pub struct CavitationDamageMaterialProperties {
    /// Yield strength (Pa)
    pub yield_strength: f64,
    /// Ultimate tensile strength (Pa)
    pub ultimate_strength: f64,
    /// Hardness (Pa)
    pub hardness: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Fatigue strength exponent
    pub fatigue_exponent: f64,
    /// Erosion resistance factor
    pub erosion_resistance: f64,
}

impl Default for CavitationDamageMaterialProperties {
    fn default() -> Self {
        Self {
            yield_strength: STAINLESS_STEEL_316_YIELD_STRENGTH,
            ultimate_strength: STAINLESS_STEEL_316_ULTIMATE_STRENGTH,
            hardness: STAINLESS_STEEL_316_HARDNESS,
            density: STAINLESS_STEEL_316_DENSITY,
            fatigue_exponent: DEFAULT_FATIGUE_EXPONENT,
            erosion_resistance: DEFAULT_EROSION_RESISTANCE,
        }
    }
}

/// Damage calculation parameters
#[derive(Debug, Clone)]
pub struct DamageParameters {
    /// Minimum impact pressure for damage (Pa)
    pub threshold_pressure: f64,
    /// Pit formation efficiency
    pub pit_efficiency: f64,
    /// Fatigue accumulation rate
    pub fatigue_rate: f64,
    /// Damage concentration factor
    pub concentration_factor: f64,
}

impl Default for DamageParameters {
    fn default() -> Self {
        Self {
            threshold_pressure: DEFAULT_THRESHOLD_PRESSURE,
            pit_efficiency: DEFAULT_PIT_EFFICIENCY,
            fatigue_rate: DEFAULT_FATIGUE_RATE,
            concentration_factor: DEFAULT_CONCENTRATION_FACTOR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::cavitation::{
        DEFAULT_CONCENTRATION_FACTOR, DEFAULT_FATIGUE_RATE, DEFAULT_PIT_EFFICIENCY,
        DEFAULT_THRESHOLD_PRESSURE,
    };

    /// `CavitationDamageMaterialProperties::default` matches the SS316 physical constants.
    #[test]
    fn material_properties_default_matches_stainless_steel_316() {
        let m = CavitationDamageMaterialProperties::default();
        assert_eq!(m.yield_strength, 290e6, "SS316 yield = 290 MPa");
        assert_eq!(m.ultimate_strength, 580e6, "SS316 UTS = 580 MPa");
        assert_eq!(m.hardness, 2.0e9, "SS316 hardness = 2 GPa");
        assert_eq!(m.density, 7850.0, "SS316 density = 7850 kg/m³");
        assert_eq!(m.fatigue_exponent, 3.0);
        assert_eq!(m.erosion_resistance, 1.0);
    }

    /// `DamageParameters::default` matches the module constants.
    #[test]
    fn damage_parameters_default_matches_module_constants() {
        let p = DamageParameters::default();
        assert_eq!(p.threshold_pressure, DEFAULT_THRESHOLD_PRESSURE);
        assert_eq!(p.pit_efficiency, DEFAULT_PIT_EFFICIENCY);
        assert_eq!(p.fatigue_rate, DEFAULT_FATIGUE_RATE);
        assert_eq!(p.concentration_factor, DEFAULT_CONCENTRATION_FACTOR);
    }
}
