//! Physical constants and standard values
//! 
//! Single Source of Truth (SSOT) for all physical constants used in simulations.
//! All values are in SI units unless otherwise specified.

use std::f64::consts::PI;

// ============================================================================
// Fundamental Physical Constants
// ============================================================================

/// Speed of light in vacuum [m/s]
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Boltzmann constant [J/K]
pub const BOLTZMANN: f64 = 1.380_649e-23;

/// Standard atmospheric pressure [Pa]
pub const ATMOSPHERIC_PRESSURE: f64 = 101_325.0;

/// Standard gravity [m/s²]
pub const GRAVITY: f64 = 9.80665;

/// Avogadro's number [1/mol]
pub const AVOGADRO: f64 = 6.022_140_76e23;

// ============================================================================
// Acoustic Properties - Water at 20°C
// ============================================================================

/// Density of water at 20°C [kg/m³]
pub const WATER_DENSITY: f64 = 998.2;

/// Speed of sound in water at 20°C [m/s]
pub const WATER_SOUND_SPEED: f64 = 1482.0;

/// Dynamic viscosity of water at 20°C [Pa·s]
pub const WATER_VISCOSITY: f64 = 1.002e-3;

/// Surface tension of water at 20°C [N/m]
pub const WATER_SURFACE_TENSION: f64 = 0.0728;

/// Specific heat capacity of water [J/(kg·K)]
pub const WATER_SPECIFIC_HEAT: f64 = 4182.0;

/// Thermal conductivity of water at 20°C [W/(m·K)]
pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.598;

/// Thermal expansion coefficient of water at 20°C [1/K]
pub const WATER_THERMAL_EXPANSION: f64 = 2.07e-4;

/// Bulk modulus of water [Pa]
pub const WATER_BULK_MODULUS: f64 = 2.2e9;

/// Nonlinearity parameter B/A for water
pub const WATER_NONLINEARITY_BA: f64 = 5.0;

// ============================================================================
// Acoustic Properties - Air at 20°C
// ============================================================================

/// Density of air at 20°C, 1 atm [kg/m³]
pub const AIR_DENSITY: f64 = 1.204;

/// Speed of sound in air at 20°C [m/s]
pub const AIR_SOUND_SPEED: f64 = 343.0;

/// Dynamic viscosity of air at 20°C [Pa·s]
pub const AIR_VISCOSITY: f64 = 1.82e-5;

/// Specific heat ratio (gamma) for air
pub const AIR_GAMMA: f64 = 1.4;

/// Polytropic index for ideal gas
pub const POLYTROPIC_INDEX_AIR: f64 = 1.4;

// ============================================================================
// Biological Tissue Properties
// ============================================================================

/// Body temperature [K]
pub const BODY_TEMPERATURE: f64 = 310.15; // 37°C

/// Blood density [kg/m³]
pub const BLOOD_DENSITY: f64 = 1060.0;

/// Blood sound speed [m/s]
pub const BLOOD_SOUND_SPEED: f64 = 1570.0;

/// Soft tissue density [kg/m³]
pub const SOFT_TISSUE_DENSITY: f64 = 1050.0;

/// Soft tissue sound speed [m/s]
pub const SOFT_TISSUE_SOUND_SPEED: f64 = 1540.0;

/// Bone density [kg/m³]
pub const BONE_DENSITY: f64 = 1900.0;

/// Bone sound speed [m/s]
pub const BONE_SOUND_SPEED: f64 = 3500.0;

/// Fat tissue density [kg/m³]
pub const FAT_DENSITY: f64 = 950.0;

/// Fat tissue sound speed [m/s]
pub const FAT_SOUND_SPEED: f64 = 1450.0;

/// Liver density [kg/m³]
pub const LIVER_DENSITY: f64 = 1060.0;

/// Liver sound speed [m/s]
pub const LIVER_SOUND_SPEED: f64 = 1595.0;

// ============================================================================
// Thermal Dose Constants
// ============================================================================

/// Reference temperature for CEM43 thermal dose [°C]
pub const CEM43_REFERENCE_TEMP: f64 = 43.0;

/// R value for CEM43 calculation above 43°C
pub const CEM43_R_ABOVE: f64 = 0.5;

/// R value for CEM43 calculation below 43°C
pub const CEM43_R_BELOW: f64 = 0.25;

/// Lethal thermal dose threshold [minutes at 43°C]
pub const THERMAL_DOSE_LETHAL: f64 = 240.0;

// ============================================================================
// Numerical Method Constants
// ============================================================================

/// CFL number for 3D FDTD stability
pub const CFL_3D_FDTD: f64 = 0.5;

/// CFL number for 2D FDTD stability
pub const CFL_2D_FDTD: f64 = 0.7;

/// CFL number for 1D FDTD stability
pub const CFL_1D_FDTD: f64 = 1.0;

/// Default reference frequency for absorption calculations [Hz]
pub const DEFAULT_REFERENCE_FREQUENCY: f64 = 1e6;

/// Minimum frequency to avoid division by zero [Hz]
pub const MIN_FREQUENCY: f64 = 1e-10;

/// Maximum reasonable frequency for medical ultrasound [Hz]
pub const MAX_MEDICAL_FREQUENCY: f64 = 20e6;

// ============================================================================
// Absorption Model Constants
// ============================================================================

/// Power law exponent for water absorption
pub const WATER_ABSORPTION_EXPONENT: f64 = 2.0;

/// Power law coefficient for water at 1 MHz [dB/(MHz^y·cm)]
pub const WATER_ABSORPTION_COEFF: f64 = 0.0022;

/// Power law exponent for blood absorption
pub const BLOOD_ABSORPTION_EXPONENT: f64 = 1.2;

/// Power law coefficient for blood at 1 MHz [dB/(MHz^y·cm)]
pub const BLOOD_ABSORPTION_COEFF: f64 = 0.18;

/// Power law exponent for soft tissue absorption
pub const TISSUE_ABSORPTION_EXPONENT: f64 = 1.1;

/// Power law coefficient for soft tissue at 1 MHz [dB/(MHz^y·cm)]
pub const TISSUE_ABSORPTION_COEFF: f64 = 0.54;

// ============================================================================
// Bubble Dynamics Constants
// ============================================================================

/// Water vapor pressure at 20°C [Pa]
pub const WATER_VAPOR_PRESSURE: f64 = 2330.0;

/// Water vapor pressure at body temperature (37°C) [Pa]
pub const BODY_VAPOR_PRESSURE: f64 = 6274.0;

/// Gas diffusion coefficient in water [m²/s]
pub const GAS_DIFFUSION_WATER: f64 = 2e-9;

/// Minimum bubble radius for stability [m]
pub const MIN_BUBBLE_RADIUS: f64 = 1e-9;

/// Maximum bubble radius before fragmentation [m]
pub const MAX_BUBBLE_RADIUS: f64 = 1e-3;

// ============================================================================
// Optical Properties
// ============================================================================

/// Refractive index of water
pub const WATER_REFRACTIVE_INDEX: f64 = 1.33;

/// Refractive index of tissue (average)
pub const TISSUE_REFRACTIVE_INDEX: f64 = 1.37;

/// Default optical absorption coefficient [1/cm]
pub const DEFAULT_OPTICAL_ABSORPTION: f64 = 0.1;

/// Default optical scattering coefficient [1/cm]
pub const DEFAULT_OPTICAL_SCATTERING: f64 = 1.0;

/// Default anisotropy factor for tissue
pub const DEFAULT_ANISOTROPY: f64 = 0.9;

// ============================================================================
// Convergence and Tolerance Constants
// ============================================================================

/// Default convergence tolerance for iterative methods
pub const CONVERGENCE_TOLERANCE: f64 = 1e-6;

/// Maximum iterations for iterative solvers
pub const MAX_ITERATIONS: usize = 1000;

/// Machine epsilon for f64
pub const EPSILON: f64 = f64::EPSILON;

/// Small value to prevent division by zero
pub const SMALL_VALUE: f64 = 1e-20;

// ============================================================================
// Grid and Discretization Constants
// ============================================================================

/// Minimum grid points per wavelength for accuracy
pub const MIN_POINTS_PER_WAVELENGTH: usize = 10;

/// Optimal grid points per wavelength for FDTD
pub const OPTIMAL_POINTS_PER_WAVELENGTH: usize = 15;

/// Maximum aspect ratio for grid cells
pub const MAX_GRID_ASPECT_RATIO: f64 = 2.0;

/// Default PML thickness in grid points
pub const PML_THICKNESS: usize = 10;

/// PML polynomial order
pub const PML_ORDER: f64 = 4.0;

/// PML reflection coefficient target
pub const PML_REFLECTION_COEFF: f64 = 1e-6;

// ============================================================================
// Validation Constants
// ============================================================================

/// Maximum acceptable relative error for validation
pub const MAX_RELATIVE_ERROR: f64 = 0.05; // 5%

/// Maximum acceptable absolute error for small values
pub const MAX_ABSOLUTE_ERROR: f64 = 1e-10;

/// Minimum signal-to-noise ratio for valid measurements [dB]
pub const MIN_SNR: f64 = 20.0;

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert dB/cm to Np/m (Nepers per meter)
#[inline]
pub const fn db_per_cm_to_np_per_m(db_per_cm: f64) -> f64 {
    db_per_cm * 100.0 / 8.686
}

/// Convert Np/m to dB/cm
#[inline]
pub const fn np_per_m_to_db_per_cm(np_per_m: f64) -> f64 {
    np_per_m * 8.686 / 100.0
}

/// Convert temperature from Celsius to Kelvin
#[inline]
pub const fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius + 273.15
}

/// Convert temperature from Kelvin to Celsius
#[inline]
pub const fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin - 273.15
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_constants_reasonable() {
        // Sanity checks for physical constants
        assert!(WATER_DENSITY > 0.0 && WATER_DENSITY < 2000.0);
        assert!(WATER_SOUND_SPEED > 1000.0 && WATER_SOUND_SPEED < 2000.0);
        assert!(ATMOSPHERIC_PRESSURE > 100_000.0 && ATMOSPHERIC_PRESSURE < 102_000.0);
        assert!(BODY_TEMPERATURE > 300.0 && BODY_TEMPERATURE < 320.0);
    }

    #[test]
    fn test_conversions() {
        // Test dB/cm to Np/m conversion
        let db_per_cm = 0.5;
        let np_per_m = db_per_cm_to_np_per_m(db_per_cm);
        assert!((np_per_m - 5.76).abs() < 0.01);

        // Test temperature conversions
        assert_eq!(celsius_to_kelvin(0.0), 273.15);
        assert_eq!(kelvin_to_celsius(273.15), 0.0);
    }
}