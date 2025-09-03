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