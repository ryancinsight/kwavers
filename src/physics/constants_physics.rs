//! Physical constants for acoustic simulations
//!
//! This module consolidates all physical constants to enforce SSOT principles
//! and eliminate magic numbers throughout the physics implementations.

use std::f64::consts::PI;

// Fundamental Physical Constants
/// Speed of sound in water at 20°C (m/s)
pub const SOUND_SPEED_WATER: f64 = 1480.0;

/// Speed of sound in soft tissue (m/s)
pub const SOUND_SPEED_TISSUE: f64 = 1540.0;

/// Density of water at 20°C (kg/m³)
pub const DENSITY_WATER: f64 = 998.0;

/// Density of soft tissue (kg/m³)
pub const DENSITY_TISSUE: f64 = 1050.0;

/// Standard atmospheric pressure (Pa)
pub const ATMOSPHERIC_PRESSURE: f64 = 101325.0;

/// Gravitational acceleration (m/s²)
pub const GRAVITY: f64 = 9.80665;

// Temperature Constants
/// Absolute zero in Celsius
pub const ABSOLUTE_ZERO_CELSIUS: f64 = -273.15;

/// Room temperature in Kelvin
pub const ROOM_TEMPERATURE_K: f64 = 293.15;

/// Body temperature in Kelvin
pub const BODY_TEMPERATURE_K: f64 = 310.15;

// Wave Propagation Constants
/// Two pi constant for frequency calculations
pub const TWO_PI: f64 = 2.0 * PI;

/// Nonlinearity parameter B/A for water
pub const NONLINEARITY_WATER: f64 = 5.0;

/// Nonlinearity parameter B/A for soft tissue
pub const NONLINEARITY_TISSUE: f64 = 6.0;

/// Absorption coefficient for water at 1 MHz (dB/cm/MHz²)
pub const ABSORPTION_WATER: f64 = 0.0022;

/// Absorption coefficient for soft tissue at 1 MHz (dB/cm/MHz^y)
pub const ABSORPTION_TISSUE: f64 = 0.75;

/// Absorption power law exponent for tissue
pub const ABSORPTION_POWER: f64 = 1.05;

// Numerical Method Constants
/// CFL condition safety factor
pub const CFL_SAFETY: f64 = 0.3;

/// Minimum points per wavelength for accurate simulation
pub const MIN_PPW: f64 = 4.0;

/// Fourth-order finite difference coefficient
pub const FD4_COEFF_0: f64 = -1.0 / 12.0;

/// Fourth-order finite difference coefficient
pub const FD4_COEFF_1: f64 = 4.0 / 3.0;

/// Fourth-order finite difference coefficient
pub const FD4_COEFF_2: f64 = -5.0 / 2.0;

// Bubble Dynamics Constants
/// Surface tension of water-air interface (N/m)
pub const SURFACE_TENSION_WATER: f64 = 0.0728;

/// Dynamic viscosity of water at 20°C (Pa·s)
pub const VISCOSITY_WATER: f64 = 1.002e-3;

/// Polytopic gas constant for air
pub const POLYTROPIC_AIR: f64 = 1.4;

/// Vapor pressure of water at 20°C (Pa)
pub const VAPOR_PRESSURE_WATER: f64 = 2338.0;

// Thermal Constants
/// Specific heat capacity of water (J/kg/K)
pub const SPECIFIC_HEAT_WATER: f64 = 4182.0;

/// Thermal conductivity of water (W/m/K)
pub const THERMAL_CONDUCTIVITY_WATER: f64 = 0.598;

/// Prandtl number for water
pub const PRANDTL_WATER: f64 = 7.0;

// Acoustic Intensity Constants
/// Reference intensity for dB calculations (W/m²)
pub const REFERENCE_INTENSITY: f64 = 1e-12;

/// Spatial peak temporal average intensity limit for diagnostic ultrasound (W/cm²)
pub const ISPTA_DIAGNOSTIC_LIMIT: f64 = 0.72;

/// Mechanical index threshold for cavitation
pub const MI_THRESHOLD: f64 = 0.7;

// Unit Conversion Constants
/// Convert dB/cm/MHz to Np/m/Hz
pub const DB_TO_NP: f64 = 0.1151;

/// Convert MHz to Hz
pub const MHZ_TO_HZ: f64 = 1e6;

/// Convert cm to m
pub const CM_TO_M: f64 = 0.01;

/// Convert μm to m
pub const UM_TO_M: f64 = 1e-6;

/// Convert bar to Pa
pub const BAR_TO_PA: f64 = 1e5;

// Validation Thresholds
/// Maximum acceptable relative error for energy conservation
pub const ENERGY_CONSERVATION_TOLERANCE: f64 = 1e-6;

/// Maximum acceptable phase velocity error
pub const PHASE_VELOCITY_TOLERANCE: f64 = 0.01;

/// Maximum acceptable amplitude decay per wavelength
pub const AMPLITUDE_DECAY_TOLERANCE: f64 = 0.05;
