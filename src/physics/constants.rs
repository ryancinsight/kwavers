//! Physical constants for acoustic simulations
//!
//! This module provides a single source of truth (SSOT) for all physical constants
//! used throughout the simulation, eliminating magic numbers and ensuring consistency.

use std::f64::consts::PI;

pub mod acoustic_parameters;

// Re-export acoustic parameters for convenience
pub use acoustic_parameters::*;

// ============================================================================
// Fundamental Physical Constants
// ============================================================================

/// Speed of sound in water at 20°C (m/s)
pub const SOUND_SPEED_WATER: f64 = 1500.0;

/// Speed of sound in soft tissue (m/s)
pub const SOUND_SPEED_TISSUE: f64 = 1540.0;

/// Speed of sound in air at 20°C (m/s)
pub const SOUND_SPEED_AIR: f64 = 343.0;

/// Density of water at 20°C (kg/m³)
pub const DENSITY_WATER: f64 = 1000.0;

/// Density of soft tissue (kg/m³)
pub const DENSITY_TISSUE: f64 = 1050.0;

/// Density of air at 20°C (kg/m³)
pub const DENSITY_AIR: f64 = 1.225;

/// Standard atmospheric pressure (Pa)
pub const ATMOSPHERIC_PRESSURE: f64 = 101325.0;

/// Vapor pressure of water at 20°C (Pa)
pub const VAPOR_PRESSURE_WATER_20C: f64 = 2339.0;

// ============================================================================
// Acoustic Properties
// ============================================================================

/// Nonlinearity parameter B/A for water
pub const B_OVER_A_WATER: f64 = 3.5;

/// Nonlinearity parameter B/A for soft tissue
pub const B_OVER_A_TISSUE: f64 = 6.0;

/// Attenuation coefficient for water (dB/cm/MHz)
pub const ATTENUATION_WATER: f64 = 0.0022;

/// Attenuation coefficient for soft tissue (dB/cm/MHz)
pub const ATTENUATION_TISSUE: f64 = 0.5;

/// Conversion factor from dB/cm to Np/m
pub const DB_CM_TO_NP_M: f64 = 100.0 / 8.686;

/// Conversion factor from dB to Np
pub const DB_TO_NP: f64 = 1.0 / 8.686;

/// Surface tension of water at 20°C (N/m)
pub const SURFACE_TENSION_WATER: f64 = 0.0728;

/// Nonlinearity coefficient for water
pub const NONLINEARITY_WATER: f64 = 3.5;

/// Absorption coefficient for tissue (dB/cm/MHz^y)
pub const ABSORPTION_TISSUE: f64 = 0.5;

/// Absorption power law exponent
pub const ABSORPTION_POWER: f64 = 1.1;

/// Conversion: `MHz` to Hz
pub const MHZ_TO_HZ: f64 = 1e6;

/// Conversion: cm to m
pub const CM_TO_M: f64 = 0.01;

/// Minimum points per wavelength for accurate simulation
pub const MIN_PPW: f64 = 10.0;

/// CFL safety factor for stability
pub const CFL_SAFETY: f64 = 0.3;

/// Mechanical Index safety threshold
pub const MI_THRESHOLD: f64 = 1.9;

// ============================================================================
// Thermal Properties
// ============================================================================

/// Specific heat capacity of water (J/(kg·K))
pub const SPECIFIC_HEAT_WATER: f64 = 4182.0;

/// Specific heat capacity of soft tissue (J/(kg·K))
pub const SPECIFIC_HEAT_TISSUE: f64 = 3600.0;

/// Thermal conductivity of water (W/(m·K))
pub const THERMAL_CONDUCTIVITY_WATER: f64 = 0.6;

/// Thermal conductivity of soft tissue (W/(m·K))
pub const THERMAL_CONDUCTIVITY_TISSUE: f64 = 0.5;

/// Thermal expansion coefficient of water at 20°C (1/K)
pub const THERMAL_EXPANSION_WATER: f64 = 2.07e-4;

/// Thermal expansion coefficient of soft tissue (1/K)
pub const THERMAL_EXPANSION_TISSUE: f64 = 3.5e-4;

/// Blood perfusion rate in tissue (kg/(m³·s))
pub const BLOOD_PERFUSION_RATE: f64 = 0.5;

/// Specific heat of blood (J/(kg·K))
pub const SPECIFIC_HEAT_BLOOD: f64 = 3840.0;

// ============================================================================
// Elastic Properties
// ============================================================================

/// Lamé first parameter for soft tissue (Pa)
pub const LAME_LAMBDA_TISSUE: f64 = 2.2e9;

/// Shear modulus for soft tissue (Pa)
pub const SHEAR_MODULUS_TISSUE: f64 = 1e6;

/// Poisson's ratio for soft tissue
pub const POISSON_RATIO_TISSUE: f64 = 0.49;

/// Young's modulus for soft tissue (Pa)
pub const YOUNGS_MODULUS_TISSUE: f64 = 3e6;

// ============================================================================
// Viscosity Properties
// ============================================================================

/// Dynamic viscosity of water at 20°C (Pa·s)
pub const VISCOSITY_WATER: f64 = 1.0e-3;

/// Bulk viscosity of water (Pa·s)
pub const BULK_VISCOSITY_WATER: f64 = 2.8e-3;

/// Shear viscosity of soft tissue (Pa·s)
pub const SHEAR_VISCOSITY_TISSUE: f64 = 0.01;

// ============================================================================
// Optical Properties
// ============================================================================

/// Optical absorption coefficient for tissue at 1064 nm (1/m)
pub const OPTICAL_ABSORPTION_TISSUE: f64 = 100.0;

/// Optical scattering coefficient for tissue (1/m)
pub const OPTICAL_SCATTERING_TISSUE: f64 = 10000.0;

/// Anisotropy factor for tissue
pub const OPTICAL_ANISOTROPY_TISSUE: f64 = 0.9;

/// Refractive index of water
pub const REFRACTIVE_INDEX_WATER: f64 = 1.33;

/// Refractive index of tissue
pub const REFRACTIVE_INDEX_TISSUE: f64 = 1.37;

// ============================================================================
// Gas Properties
// ============================================================================

/// Gas diffusion coefficient (O2 in water) (m²/s)
pub const GAS_DIFFUSION_O2_WATER: f64 = 2.0e-9;

/// Henry's law constant for O2 in water at 20°C
pub const HENRY_CONSTANT_O2: f64 = 1.3e-3;

// ============================================================================
// Numerical Constants
// ============================================================================

/// CFL number for stability
pub const CFL_NUMBER: f64 = 0.3;

/// Minimum wavelengths per grid spacing
pub const MIN_WAVELENGTHS_PER_GRID: f64 = 10.0;

/// Maximum allowed pressure for stability (Pa)
pub const MAX_PRESSURE_LIMIT: f64 = 1e10;

/// Minimum density to avoid division by zero (kg/m³)
pub const MIN_DENSITY: f64 = 0.1;

/// Small value for numerical stability
pub const EPSILON: f64 = 1e-15;

/// Energy conservation tolerance for validation
pub const ENERGY_CONSERVATION_TOLERANCE: f64 = 1e-6;

// ============================================================================
// Conversion Functions
// ============================================================================

/// Convert dB/cm/MHz to Np/m at given frequency
#[must_use]
pub fn db_cm_mhz_to_np_m(alpha_db_cm_mhz: f64, frequency_hz: f64) -> f64 {
    let frequency_mhz = frequency_hz / 1e6;
    alpha_db_cm_mhz * frequency_mhz * DB_CM_TO_NP_M
}

/// Convert Np/m to dB/cm/MHz at given frequency
#[must_use]
pub fn np_m_to_db_cm_mhz(alpha_np_m: f64, frequency_hz: f64) -> f64 {
    let frequency_mhz = frequency_hz / 1e6;
    alpha_np_m / (frequency_mhz * DB_CM_TO_NP_M)
}

/// Calculate wavelength from frequency and sound speed
#[must_use]
pub fn wavelength(frequency: f64, sound_speed: f64) -> f64 {
    sound_speed / frequency
}

/// Calculate wavenumber
#[must_use]
pub fn wavenumber(frequency: f64, sound_speed: f64) -> f64 {
    2.0 * PI * frequency / sound_speed
}

/// Convert Kelvin to Celsius
#[must_use]
pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin - 273.15
}

/// Convert Celsius to Kelvin
#[must_use]
pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius + 273.15
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_consistency() {
        let alpha_db = 0.5; // dB/cm/MHz
        let freq = 1e6; // 1 MHz

        let alpha_np = db_cm_mhz_to_np_m(alpha_db, freq);
        let alpha_db_back = np_m_to_db_cm_mhz(alpha_np, freq);

        assert!((alpha_db - alpha_db_back).abs() < 1e-10);
    }

    #[test]
    fn test_wavelength_calculation() {
        let freq = 1e6; // 1 MHz
        let lambda = wavelength(freq, SOUND_SPEED_WATER);
        assert!((lambda - 0.0015).abs() < 1e-6); // 1.5 mm
    }
}
