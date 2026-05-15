//! Physical constants for CT-derived skull acoustic properties.

use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

/// Hounsfield unit of pure water (calibration reference).
pub const HU_WATER: f64 = 0.0;
/// Hounsfield unit of fully mineralised cortical bone.
pub const HU_CORTICAL: f64 = 1000.0;
/// Sound speed in water at 20 °C [m/s] — k-Wave simulation default.
pub const C_WATER: f64 = SOUND_SPEED_WATER_SIM;
/// Density of water at 20 °C [kg/m³] — nominal round-number default.
pub const RHO_WATER: f64 = DENSITY_WATER_NOMINAL;
/// Water attenuation at clinical frequencies [Np/m/MHz].
pub const ALPHA_WATER: f64 = 0.002;
