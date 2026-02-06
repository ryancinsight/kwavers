//! Physics constants re-exports
//!
//! This module re-exports physical constants from the core::constants module
//! to maintain backward compatibility with existing code that imports from physics::constants.
//!
//! **SSOT**: All constants are defined in `crate::core::constants`.
//! This module only re-exports them for convenience.

// Re-export all core constants at the physics::constants level
pub use crate::core::constants::*;

// Explicit re-exports for commonly used constants to maintain compatibility
pub use crate::core::constants::acoustic_parameters::REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ;
pub use crate::core::constants::fundamental::{
    DENSITY_TISSUE, DENSITY_WATER, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER,
};
pub use crate::core::constants::numerical::{B_OVER_A_DIVISOR, NONLINEARITY_COEFFICIENT_OFFSET};
pub use crate::core::constants::water::{
    BULK_MODULUS_WATER, C_WATER, SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER,
};
