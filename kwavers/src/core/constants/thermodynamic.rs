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

/// Thermal conductivity of air (W/(m·K))
pub const THERMAL_CONDUCTIVITY_AIR: f64 = 0.026;

// ============================================================================
// Van der Waals Constants
// ============================================================================
// Format: (a in bar·L²/mol², b in L/mol)
// References: CRC Handbook of Chemistry and Physics

/// Van der Waals arbitrary constants for Air (a, b)
pub const VAN_DER_WAALS_AIR: (f64, f64) = (1.37, 0.0387);
/// Van der Waals constants for Argon (a, b)
pub const VAN_DER_WAALS_ARGON: (f64, f64) = (1.355, 0.0320);
/// Van der Waals constants for Xenon (a, b)
pub const VAN_DER_WAALS_XENON: (f64, f64) = (4.250, 0.0510);
/// Van der Waals constants for Nitrogen (a, b)
pub const VAN_DER_WAALS_NITROGEN: (f64, f64) = (1.370, 0.0387);
/// Van der Waals constants for Oxygen (a, b)
pub const VAN_DER_WAALS_OXYGEN: (f64, f64) = (1.382, 0.0319);

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

/// Emissivity of water vapor in collapsing acoustic cavitation bubbles (dimensionless)
///
/// Value: 0.1 — lower bound for hot-water-vapor emissivity used in acoustic cavitation modelling.
///
/// At extreme bubble collapse temperatures (T > 10,000 K) the vapor approximates a grey-body
/// radiator. Measured emissivity for steam at high temperatures spans 0.1–0.3; 0.1 is a
/// conservative estimate consistent with single-bubble sonoluminescence observations where
/// radiative losses are secondary to conductive cooling.
///
/// Reference: Suslick, K.S. & Flannigan, D.J. (2008). "Inside a collapsing bubble:
/// sonoluminescence and the conditions during cavitation." Annu. Rev. Phys. Chem. 59:659–683.
pub const EMISSIVITY_VAPOR: f64 = 0.1;

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

// Gas constant is available in fundamental.rs
pub use super::fundamental::GAS_CONSTANT as R_GAS;
