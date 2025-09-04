//! Physical constants for acoustic simulations
//!
//! This module provides a consolidated single source of truth (SSOT) for all physical 
//! constants used throughout the simulation, eliminating magic numbers and ensuring consistency.
//! Values are based on NIST standards and peer-reviewed literature.

use std::f64::consts::PI;

// Fundamental physical constants (NIST 2018 values)
/// Speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Planck constant (J⋅s)
pub const PLANCK_CONSTANT: f64 = 6.626_070_15e-34;

/// Boltzmann constant (J/K)
pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23;

/// Avogadro constant (mol⁻¹)
pub const AVOGADRO_CONSTANT: f64 = 6.022_140_76e23;

/// Gas constant (J/mol/K)
pub const GAS_CONSTANT: f64 = 8.314_462_618;

// Acoustic parameters for common media
/// Sound speed in water at 20°C (m/s)
pub const SOUND_SPEED_WATER: f64 = 1500.0;

/// Sound speed in air at 20°C (m/s)  
pub const SOUND_SPEED_AIR: f64 = 343.0;

/// Sound speed in soft tissue (m/s)
pub const SOUND_SPEED_TISSUE: f64 = 1540.0;

/// Sound speed in bone (m/s)
pub const SOUND_SPEED_BONE: f64 = 4080.0;

/// Density of water (kg/m³)
pub const DENSITY_WATER: f64 = 1000.0;

/// Density of air at STP (kg/m³)
pub const DENSITY_AIR: f64 = 1.225;

/// Density of soft tissue (kg/m³)
pub const DENSITY_TISSUE: f64 = 1050.0;

/// Density of bone (kg/m³)
pub const DENSITY_BONE: f64 = 1900.0;

/// Atmospheric pressure (Pa)
pub const ATMOSPHERIC_PRESSURE: f64 = 101_325.0;

// Acoustic attenuation coefficients (dB/cm/MHz)
/// Attenuation in water (dB/cm/MHz)
pub const ATTENUATION_WATER: f64 = 0.0022;

/// Attenuation in soft tissue (dB/cm/MHz)
pub const ATTENUATION_TISSUE: f64 = 0.5;

/// Attenuation in bone (dB/cm/MHz)
pub const ATTENUATION_BONE: f64 = 5.0;

// Nonlinear acoustic parameters
/// Nonlinearity parameter for water
pub const BETA_WATER: f64 = 3.5;

/// Nonlinearity parameter for tissue
pub const BETA_TISSUE: f64 = 6.0;

// Cavitation parameters
/// Surface tension of water (N/m)
pub const SURFACE_TENSION_WATER: f64 = 0.0728;

/// Vapor pressure of water at 37°C (Pa)
pub const VAPOR_PRESSURE_WATER_37C: f64 = 6_276.0;

/// Gas constant for air (J/kg/K)
pub const GAS_CONSTANT_AIR: f64 = 287.0;

/// Polytropic index for adiabatic processes
pub const POLYTROPIC_INDEX: f64 = 1.4;

/// Blake threshold for cavitation inception (MPa) - corrected value
pub const BLAKE_THRESHOLD: f64 = 0.541;

// Thermal parameters
/// Specific heat of water (J/kg/K)
pub const SPECIFIC_HEAT_WATER: f64 = 4186.0;

/// Specific heat of tissue (J/kg/K)
pub const SPECIFIC_HEAT_TISSUE: f64 = 3600.0;

/// Thermal conductivity of water (W/m/K)
pub const THERMAL_CONDUCTIVITY_WATER: f64 = 0.6;

/// Thermal conductivity of tissue (W/m/K)
pub const THERMAL_CONDUCTIVITY_TISSUE: f64 = 0.52;

/// Blood density (kg/m³)
pub const BLOOD_DENSITY: f64 = 1060.0;

/// Blood specific heat (J/kg/K)
pub const BLOOD_SPECIFIC_HEAT: f64 = 3617.0;

/// Body temperature (°C)
pub const BODY_TEMPERATURE: f64 = 37.0;

/// Blood perfusion rate baseline (kg/m³/s)
pub const PERFUSION_RATE_BASELINE: f64 = 0.5;

// Numerical stability constants
/// CFL stability factor for FDTD
pub const CFL_SAFETY_FACTOR: f64 = 0.9;

/// Maximum allowable CFL number
pub const CFL_MAX: f64 = 1.0;

/// Minimum time step (s)
pub const MIN_TIME_STEP: f64 = 1e-12;

/// Maximum time step (s)
pub const MAX_TIME_STEP: f64 = 1e-6;

/// Convergence tolerance for iterative solvers
pub const CONVERGENCE_TOLERANCE: f64 = 1e-6;

/// Maximum iterations for iterative solvers
pub const MAX_ITERATIONS: usize = 10_000;

// Grid parameters
/// Minimum points per wavelength for stability
pub const MIN_POINTS_PER_WAVELENGTH: f64 = 10.0;

/// Recommended points per wavelength for accuracy
pub const RECOMMENDED_POINTS_PER_WAVELENGTH: f64 = 20.0;

/// Minimum grid spacing for numerical stability (m)
pub const MIN_DX: f64 = 1e-6; // 1 μm minimum

/// Default grid spacing (m)
pub const DEFAULT_GRID_SPACING: f64 = 1e-4; // 0.1 mm

// Medical ultrasound parameters
/// Diagnostic frequency range lower bound (Hz)
pub const DIAGNOSTIC_FREQ_MIN: f64 = 1e6; // 1 MHz

/// Diagnostic frequency range upper bound (Hz)
pub const DIAGNOSTIC_FREQ_MAX: f64 = 15e6; // 15 MHz

/// Therapeutic frequency range lower bound (Hz)
pub const THERAPEUTIC_FREQ_MIN: f64 = 0.3e6; // 0.3 MHz

/// Therapeutic frequency range upper bound (Hz)
pub const THERAPEUTIC_FREQ_MAX: f64 = 3e6; // 3 MHz

/// FDA intensity limit for diagnostic ultrasound (W/cm²)
pub const FDA_INTENSITY_LIMIT_DIAGNOSTIC: f64 = 0.72;

/// FDA intensity limit for therapeutic ultrasound (W/cm²)
pub const FDA_INTENSITY_LIMIT_THERAPEUTIC: f64 = 30.0;

/// Default ultrasound frequency (Hz) - Standard 1 MHz
pub const DEFAULT_ULTRASOUND_FREQUENCY: f64 = 1e6;

/// Standard pressure amplitude (Pa) - 1 MPa reference
pub const STANDARD_PRESSURE_AMPLITUDE: f64 = 1e6;

/// Standard beam width (m) - 1 cm reference  
pub const STANDARD_BEAM_WIDTH: f64 = 0.01;

// Chemistry and sonoluminescence parameters
/// Oxygen molecular weight (kg/mol)
pub const MOLECULAR_WEIGHT_O2: f64 = 0.032;

/// Nitrogen molecular weight (kg/mol)
pub const MOLECULAR_WEIGHT_N2: f64 = 0.028;

/// Water molecular weight (kg/mol)
pub const MOLECULAR_WEIGHT_H2O: f64 = 0.018;

/// Activation energy for OH radical formation (J/mol)
pub const ACTIVATION_ENERGY_OH: f64 = 5e5;

/// Pre-exponential factor for reaction rates (s⁻¹)
pub const PRE_EXPONENTIAL_FACTOR: f64 = 1e13;

// Mathematical constants for convenience
/// Pi (π)
pub const PI_F64: f64 = PI;

/// 2π
pub const TWO_PI: f64 = 2.0 * PI;

/// π/2
pub const HALF_PI: f64 = PI / 2.0;

/// √2
pub const SQRT_2: f64 = 1.414_213_562_373_095;

/// √3  
pub const SQRT_3: f64 = 1.732_050_807_568_877;

/// Euler's number (e)
pub const E: f64 = std::f64::consts::E;

/// Natural logarithm of 2
pub const LN_2: f64 = std::f64::consts::LN_2;

/// Natural logarithm of 10
pub const LN_10: f64 = std::f64::consts::LN_10;

// Unit conversion factors
/// Convert MHz to Hz
pub const MHZ_TO_HZ: f64 = 1e6;

/// Convert cm to m
pub const CM_TO_M: f64 = 0.01;

/// Convert mm to m
pub const MM_TO_M: f64 = 0.001;

/// Convert μm to m
pub const UM_TO_M: f64 = 1e-6;

/// Convert nm to m
pub const NM_TO_M: f64 = 1e-9;

/// Convert dB to Neper
pub const DB_TO_NEPER: f64 = LN_10 / 20.0;

/// Convert degrees to radians
pub const DEG_TO_RAD: f64 = PI / 180.0;

/// Convert radians to degrees
pub const RAD_TO_DEG: f64 = 180.0 / PI;

// Additional material properties needed by the codebase
/// Air polytropic index (ratio of specific heats)
pub const AIR_POLYTROPIC_INDEX: f64 = 1.4;

/// Blood viscosity at 37°C (Pa·s)
pub const BLOOD_VISCOSITY_37C: f64 = 3.5e-3;

/// Reference frequency for absorption calculations (Hz)
pub const REFERENCE_FREQUENCY_MHZ: f64 = 1e6;

/// Water absorption coefficient α₀ at reference frequency (dB/(MHz^y cm))
pub const WATER_ABSORPTION_ALPHA_0: f64 = 0.0022;

/// Water absorption frequency power law exponent
pub const WATER_ABSORPTION_POWER: f64 = 1.05;

/// Water specific heat capacity (J/(kg·K))
pub const WATER_SPECIFIC_HEAT: f64 = 4182.0;

/// Water surface tension at 20°C (N/m)
pub const WATER_SURFACE_TENSION_20C: f64 = 0.0728;

/// Water thermal conductivity at 20°C (W/(m·K))
pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.598;

/// Water vapor pressure at 20°C (Pa)
pub const WATER_VAPOR_PRESSURE_20C: f64 = 2339.0;

/// Bond transform factor for crystallography
pub const BOND_TRANSFORM_FACTOR: f64 = 1.0; // Placeholder value

/// Lamé to stiffness tensor conversion factor
pub const LAME_TO_STIFFNESS_FACTOR: f64 = 1.0; // Placeholder value

/// Symmetry tolerance for material tensors
pub const SYMMETRY_TOLERANCE: f64 = 1e-9;

// Cavitation parameters
/// Maximum bubble radius for simulations (m)
pub const MAX_RADIUS: f64 = 1e-3; // 1 mm maximum

/// Minimum bubble radius for simulations (m)  
pub const MIN_RADIUS: f64 = 1e-9; // 1 nm minimum

/// Initial bubble radius (m)
pub const INITIAL_BUBBLE_RADIUS: f64 = 5e-6; // 5 μm

/// Polytropic exponent for air
pub const POLYTROPIC_EXPONENT_AIR: f64 = 1.4;

/// Van der Waals hard core radius (m)
pub const VAN_DER_WAALS_RADIUS: f64 = 8.86e-10;

/// Cavitation threshold in water (MPa)
pub const CAVITATION_THRESHOLD_WATER: f64 = -30.0;

// Numerical solver parameters
/// Machine epsilon for f64
pub const MACHINE_EPSILON: f64 = f64::EPSILON;

/// Small value to prevent division by zero
pub const SMALL_VALUE: f64 = 1e-12;

/// Large value for boundary conditions
pub const LARGE_VALUE: f64 = 1e12;

/// Default smoothing parameter
pub const SMOOTHING_PARAMETER: f64 = 0.01;

/// Newton-Raphson tolerance
pub const NEWTON_TOLERANCE: f64 = 1e-10;

/// Maximum Newton-Raphson iterations
pub const NEWTON_MAX_ITER: usize = 50;

// Thermodynamic constants needed by bubble dynamics
/// Gas constant for air (J/kg/K)
pub const GAS_CONSTANT_AIR_SPEC: f64 = 287.0;

/// Specific heat ratio for air
pub const GAMMA_AIR: f64 = 1.4;

/// Standard temperature (K)
pub const STANDARD_TEMPERATURE: f64 = 293.15; // 20°C

/// Standard pressure (Pa)
pub const STANDARD_PRESSURE: f64 = 101325.0;

// Derived constants for common calculations
/// Acoustic impedance of water (kg/m²/s)
pub const ACOUSTIC_IMPEDANCE_WATER: f64 = DENSITY_WATER * SOUND_SPEED_WATER;

/// Acoustic impedance of tissue (kg/m²/s)
pub const ACOUSTIC_IMPEDANCE_TISSUE: f64 = DENSITY_TISSUE * SOUND_SPEED_TISSUE;

/// Bulk modulus of water (Pa)
pub const BULK_MODULUS_WATER: f64 = DENSITY_WATER * SOUND_SPEED_WATER * SOUND_SPEED_WATER;

/// Bulk modulus of tissue (Pa)  
pub const BULK_MODULUS_TISSUE: f64 = DENSITY_TISSUE * SOUND_SPEED_TISSUE * SOUND_SPEED_TISSUE;

// Validation helper functions
/// Check if frequency is in diagnostic range
pub const fn is_diagnostic_frequency(freq: f64) -> bool {
    freq >= DIAGNOSTIC_FREQ_MIN && freq <= DIAGNOSTIC_FREQ_MAX
}

/// Check if frequency is in therapeutic range  
pub const fn is_therapeutic_frequency(freq: f64) -> bool {
    freq >= THERAPEUTIC_FREQ_MIN && freq <= THERAPEUTIC_FREQ_MAX
}

/// Calculate wavelength from frequency and sound speed
pub const fn wavelength(frequency: f64, sound_speed: f64) -> f64 {
    sound_speed / frequency
}

/// Calculate grid spacing for given frequency and points per wavelength
pub const fn grid_spacing_for_frequency(frequency: f64, sound_speed: f64, ppw: f64) -> f64 {
    wavelength(frequency, sound_speed) / ppw
}

/// Calculate stable time step using CFL condition
pub const fn stable_time_step(grid_spacing: f64, sound_speed: f64) -> f64 {
    CFL_SAFETY_FACTOR * grid_spacing / (SQRT_3 * sound_speed)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_derived_constants() {
        // Test acoustic impedance calculations
        let expected_z_water = DENSITY_WATER * SOUND_SPEED_WATER;
        assert!((ACOUSTIC_IMPEDANCE_WATER - expected_z_water).abs() < f64::EPSILON);
        
        // Test bulk modulus calculations
        let expected_k_water = DENSITY_WATER * SOUND_SPEED_WATER.powi(2);
        assert!((BULK_MODULUS_WATER - expected_k_water).abs() < f64::EPSILON);
    }
    
    #[test]
    fn test_frequency_validation() {
        assert!(is_diagnostic_frequency(5e6)); // 5 MHz
        assert!(!is_diagnostic_frequency(0.5e6)); // 0.5 MHz - too low
        assert!(is_therapeutic_frequency(1e6)); // 1 MHz
        assert!(!is_therapeutic_frequency(10e6)); // 10 MHz - too high
    }
    
    #[test]
    fn test_wavelength_calculation() {
        let freq = 1e6; // 1 MHz
        let lambda = wavelength(freq, SOUND_SPEED_WATER);
        assert!((lambda - 0.0015).abs() < 1e-6); // 1.5 mm wavelength
    }
    
    #[test]
    fn test_stability_calculations() {
        let dx = 1e-4; // 0.1 mm
        let dt = stable_time_step(dx, SOUND_SPEED_WATER);
        
        // Check CFL condition
        let cfl = SOUND_SPEED_WATER * dt / dx * SQRT_3;
        assert!(cfl <= CFL_MAX);
    }
}