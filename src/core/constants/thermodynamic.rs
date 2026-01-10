//! Thermodynamic constants

/// Room temperature in Kelvin
pub const ROOM_TEMPERATURE_K: f64 = 293.15;

/// Body temperature in Kelvin
pub const BODY_TEMPERATURE_K: f64 = 310.15;

/// Room temperature in Celsius
pub const ROOM_TEMPERATURE_C: f64 = 20.0;

/// Body temperature in Celsius
pub const BODY_TEMPERATURE_C: f64 = 37.0;

/// Absolute zero in Celsius
pub const ABSOLUTE_ZERO_C: f64 = -273.15;

/// Triple point of water temperature (K)
pub const WATER_TRIPLE_POINT_K: f64 = 273.16;

/// Critical temperature of water (K)
pub const WATER_CRITICAL_TEMP_K: f64 = 647.096;

/// Critical pressure of water (Pa)
pub const WATER_CRITICAL_PRESSURE: f64 = 22.064e6;

/// Specific heat capacity of water at 20°C (J/(kg·K))
pub const SPECIFIC_HEAT_WATER: f64 = 4186.0;

/// Specific heat capacity of tissue (J/(kg·K))
pub const SPECIFIC_HEAT_TISSUE: f64 = 3600.0;

/// Thermal conductivity of water at 20°C (W/(m·K))
pub const THERMAL_CONDUCTIVITY_WATER: f64 = 0.598;

/// Thermal conductivity of tissue (W/(m·K))
pub const THERMAL_CONDUCTIVITY_TISSUE: f64 = 0.5;

/// Thermal diffusivity of water (m²/s)
pub const THERMAL_DIFFUSIVITY_WATER: f64 = 1.43e-7;

/// Thermal diffusivity of tissue (m²/s)
pub const THERMAL_DIFFUSIVITY_TISSUE: f64 = 1.36e-7;

/// Molar mass of water (kg/mol)
pub const M_WATER: f64 = 0.018015;

// ============================================================================
// Heat Transfer Constants
// ============================================================================

/// Nusselt number constant term
pub const NUSSELT_CONSTANT: f64 = 2.0;

/// Nusselt number Peclet coefficient
pub const NUSSELT_PECLET_COEFF: f64 = 0.45;

/// Nusselt number Peclet exponent
pub const NUSSELT_PECLET_EXPONENT: f64 = 0.5;

/// Sherwood number Peclet exponent
pub const SHERWOOD_PECLET_EXPONENT: f64 = 0.33;

/// Ambient temperature (K)
pub const T_AMBIENT: f64 = 293.15;

/// Vapor diffusion coefficient in air (m²/s)
pub const VAPOR_DIFFUSION_COEFFICIENT: f64 = 2.5e-5;

// ============================================================================
// Chemical Reaction Constants
// ============================================================================

/// Reaction reference temperature (K)
pub const REACTION_REFERENCE_TEMPERATURE: f64 = 298.15;

/// Secondary reaction rate constant (1/s)
pub const SECONDARY_REACTION_RATE: f64 = 1e-3;

/// Sonochemistry base reaction rate (1/s)
pub const SONOCHEMISTRY_BASE_RATE: f64 = 1e-2;

// ============================================================================
// Water Properties at Specific Conditions
// ============================================================================

/// Heat of vaporization of water at 100°C (J/kg)
pub const H_VAP_WATER_100C: f64 = 2.257e6;

/// Atmospheric pressure (Pa)
pub const P_ATM: f64 = 101325.0;

/// Critical pressure of water (Pa)
pub const P_CRITICAL_WATER: f64 = 22.064e6;

/// Triple point pressure of water (Pa)
pub const P_TRIPLE_WATER: f64 = 611.657;

/// Boiling temperature of water at 1 atm (K)
pub const T_BOILING_WATER: f64 = 373.15;

/// Critical temperature of water (K)
pub const T_CRITICAL_WATER: f64 = 647.096;

/// Triple point temperature of water (K)
pub const T_TRIPLE_WATER: f64 = 273.16;

/// Latent heat of vaporization of water (J/kg)
pub const WATER_LATENT_HEAT_VAPORIZATION: f64 = 2.45e6;

/// Convert temperature from Kelvin to Celsius
#[inline]
pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin + ABSOLUTE_ZERO_C
}

/// Convert temperature from Celsius to Kelvin
#[inline]
pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius - ABSOLUTE_ZERO_C
}

// Re-export R_GAS for convenience
pub use crate::physics::constants::GAS_CONSTANT as R_GAS;
