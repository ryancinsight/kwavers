//! Physical constants for KZK equation simulations
//!
//! Provides named constants to eliminate magic numbers

use crate::core::constants::SOUND_SPEED_WATER;

/// Default test frequency: 1 `MHz`
pub const DEFAULT_FREQUENCY: f64 = 1e6;

/// Default wavelength in water at 1 `MHz`
pub const DEFAULT_WAVELENGTH: f64 = SOUND_SPEED_WATER / DEFAULT_FREQUENCY;

/// Default spatial step for KZK simulations
pub const DEFAULT_DX: f64 = 0.25e-3; // 0.25 mm (6 points per wavelength at 1 MHz)

/// Default axial step for KZK simulations  
pub const DEFAULT_DZ: f64 = 1e-3; // 1 mm

/// Default temporal step
pub const DEFAULT_DT: f64 = 1e-6; // 1 μs

/// Default beam waist for tests: 5 mm
pub const DEFAULT_BEAM_WAIST: f64 = 5e-3;

/// Default source amplitude: 1 `MPa`
pub const DEFAULT_AMPLITUDE: f64 = 1e6;

/// Threshold factor for beam width measurement (1/e²)
pub const BEAM_WIDTH_THRESHOLD_FACTOR: f64 = std::f64::consts::E * std::f64::consts::E;

/// Default grid size for tests
pub const DEFAULT_GRID_SIZE: usize = 128;

/// Nonlinearity parameter B/A for water
pub const B_OVER_A_WATER: f64 = 5.0;

/// Nonlinearity parameter B/A for soft tissue
pub const B_OVER_A_TISSUE: f64 = 6.0;

/// Attenuation coefficient for water at 1 `MHz` [dB/(MHz^y cm)]
pub const ALPHA_WATER: f64 = 0.0022;

/// Attenuation power law exponent for water
pub const ALPHA_POWER_WATER: f64 = 2.0;

/// Attenuation coefficient for soft tissue at 1 `MHz` [dB/(MHz^y cm)]
pub const ALPHA_TISSUE: f64 = 0.5;

/// Attenuation power law exponent for soft tissue
pub const ALPHA_POWER_TISSUE: f64 = 1.1;
