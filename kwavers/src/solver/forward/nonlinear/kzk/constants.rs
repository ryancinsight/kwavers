//! Physical constants for KZK equation simulations
//!
//! All tissue and medium property values delegate to the canonical SSOT at
//! `core::constants`.  No numerical literals are defined locally.

use crate::core::constants::acoustic_parameters::WATER_ABSORPTION_ALPHA_0;
use crate::core::constants::fundamental::{
    ACOUSTIC_ABSORPTION_TISSUE, B_OVER_A_SOFT_TISSUE, B_OVER_A_WATER,
    SOFT_TISSUE_ABSORPTION_POWER_Y, WATER_ABSORPTION_POWER_Y,
};
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

// ── Acoustic medium properties — thin wrappers delegating to SSOT constants ──

/// Nonlinearity parameter B/A for water at 20°C (dimensionless).
///
/// Delegates to `fundamental::B_OVER_A_WATER` = 5.2 (Duck 1990 Table 4.16).
pub const B_OVER_A_WATER_KZK: f64 = B_OVER_A_WATER;

/// Nonlinearity parameter B/A for generic soft tissue (dimensionless).
///
/// Delegates to `fundamental::B_OVER_A_SOFT_TISSUE` = 6.5 (Duck 1990 mean).
pub const B_OVER_A_TISSUE: f64 = B_OVER_A_SOFT_TISSUE;

/// Attenuation coefficient for water at 1 MHz [dB/(MHz^y cm)].
///
/// Delegates to `acoustic_parameters::WATER_ABSORPTION_ALPHA_0` = 0.0022.
pub const ALPHA_WATER: f64 = WATER_ABSORPTION_ALPHA_0;

/// Attenuation power-law exponent for water (quadratic f² law, y = 2.0).
///
/// Delegates to `fundamental::WATER_ABSORPTION_POWER_Y`.
pub const ALPHA_POWER_WATER: f64 = WATER_ABSORPTION_POWER_Y;

/// Attenuation coefficient for soft tissue at 1 MHz [dB/(MHz^y cm)].
///
/// Delegates to `fundamental::ACOUSTIC_ABSORPTION_TISSUE` = 0.5.
pub const ALPHA_TISSUE: f64 = ACOUSTIC_ABSORPTION_TISSUE;

/// Attenuation power-law exponent for soft tissue (y ≈ 1.1).
///
/// Delegates to `fundamental::SOFT_TISSUE_ABSORPTION_POWER_Y`.
pub const ALPHA_POWER_TISSUE: f64 = SOFT_TISSUE_ABSORPTION_POWER_Y;
