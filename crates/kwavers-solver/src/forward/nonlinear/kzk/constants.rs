//! Physical constants for KZK equation simulations
//!
//! All tissue and medium property values delegate to the canonical SSOT at
//! `core::constants`.  No numerical literals are defined locally.

use kwavers_core::constants::acoustic_parameters::REFERENCE_FREQUENCY_HZ;
use kwavers_core::constants::SOUND_SPEED_WATER;

/// Default test frequency: 1 `MHz` — delegates to `acoustic_parameters::REFERENCE_FREQUENCY_HZ`.
pub const DEFAULT_FREQUENCY: f64 = REFERENCE_FREQUENCY_HZ;

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

/// Threshold factor for beam width measurement (1/e²)
pub const BEAM_WIDTH_THRESHOLD_FACTOR: f64 = std::f64::consts::E * std::f64::consts::E;

/// Default grid size for tests
pub const DEFAULT_GRID_SIZE: usize = 128;
