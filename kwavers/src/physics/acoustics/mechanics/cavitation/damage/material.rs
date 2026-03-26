//! Material properties and damage parameters for cavitation modeling

// Material constants for stainless steel 316
const STAINLESS_STEEL_316_YIELD_STRENGTH: f64 = 290e6; // Pa
const STAINLESS_STEEL_316_ULTIMATE_STRENGTH: f64 = 580e6; // Pa
const STAINLESS_STEEL_316_HARDNESS: f64 = 2.0e9; // Pa
const STAINLESS_STEEL_316_DENSITY: f64 = 7850.0; // kg/m³
const DEFAULT_FATIGUE_EXPONENT: f64 = 3.0;
const DEFAULT_EROSION_RESISTANCE: f64 = 1.0;

use crate::core::constants::cavitation::{
    DEFAULT_CONCENTRATION_FACTOR, DEFAULT_FATIGUE_RATE, DEFAULT_PIT_EFFICIENCY,
    DEFAULT_THRESHOLD_PRESSURE,
};

/// Material properties for damage calculation
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    /// Yield strength [Pa]
    pub yield_strength: f64,
    /// Ultimate tensile strength [Pa]
    pub ultimate_strength: f64,
    /// Hardness [Pa]
    pub hardness: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Fatigue strength exponent
    pub fatigue_exponent: f64,
    /// Erosion resistance factor
    pub erosion_resistance: f64,
}

impl Default for MaterialProperties {
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
    /// Minimum impact pressure for damage [Pa]
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
