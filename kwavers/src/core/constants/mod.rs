//! Physical constants for acoustic simulations
//!
//! This module provides a single source of truth (SSOT) for all physical constants
//! used throughout the simulation, eliminating magic numbers and ensuring consistency.

pub mod acoustic_parameters;
pub mod cavitation;
pub mod chemistry;
pub mod fundamental;
pub mod hounsfield;
pub mod medical;
pub mod numerical;
pub mod optical;
pub mod state_dependent;
pub mod thermodynamic;
pub mod water;

// ============================================================================
// EXPLICIT RE-EXPORTS (Core Physical Constants API)
// ============================================================================
// Re-export the most commonly used constants. For specialized constants,
// import directly from submodules: `use crate::core::constants::cavitation::BLAKE_THRESHOLD;`

// Fundamental physical constants
pub use fundamental::{
    ATMOSPHERIC_PRESSURE, AVOGADRO, BOLTZMANN, BOND_TRANSFORM_FACTOR, DENSITY_AIR, DENSITY_TISSUE,
    GAS_CONSTANT, GRAVITY, LAME_TO_STIFFNESS_FACTOR, PLANCK, SOUND_SPEED_AIR, SOUND_SPEED_TISSUE,
    SOUND_SPEED_WATER, SPEED_OF_LIGHT, SYMMETRY_TOLERANCE,
};

// Water properties (most common medium)
pub use cavitation::{SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER};
pub use fundamental::{BULK_MODULUS_WATER, C_WATER, DENSITY_WATER};

// Cavitation control limits
pub use cavitation::{MAX_DUTY_CYCLE, MIN_DUTY_CYCLE};

// Acoustic parameters (most commonly used)
pub use acoustic_parameters::{
    AIR_POLYTROPIC_INDEX, BLOOD_VISCOSITY_37C, DIAGNOSTIC_FREQ_MAX, DIAGNOSTIC_FREQ_MIN,
    REFERENCE_FREQUENCY_MHZ, SAMPLING_FREQUENCY_DEFAULT, WATER_ABSORPTION_ALPHA_0,
    WATER_ABSORPTION_POWER, WATER_NONLINEARITY_B_A, WATER_SPECIFIC_HEAT, WATER_SURFACE_TENSION_20C,
    WATER_THERMAL_CONDUCTIVITY, WATER_VAPOR_PRESSURE_20C, WATER_VISCOSITY_20C,
};

// Thermodynamic constants
pub use thermodynamic::{
    BODY_TEMPERATURE_C, BODY_TEMPERATURE_K, H_VAP_WATER_100C, M_WATER, P_ATM, P_CRITICAL_WATER,
    P_TRIPLE_WATER, ROOM_TEMPERATURE_C, ROOM_TEMPERATURE_K, SPECIFIC_HEAT_WATER,
    THERMAL_CONDUCTIVITY_WATER, T_BOILING_WATER, T_CRITICAL_WATER, T_TRIPLE_WATER,
};

// Medical constants (FDA limits)
pub use medical::{FDA_DERATING_FACTOR, ISPPA_LIMIT, ISPTA_LIMIT, THERMAL_DOSE_THRESHOLD};

// Numerical constants
pub use numerical::{CFL_DEFAULT, CFL_SAFETY_FACTOR, EPSILON, SMALL_VALUE, SOLVER_TOLERANCE};

// Backward compatibility aliases
pub use fundamental::GAS_CONSTANT as R_GAS;

// State-dependent constants (NEW: addresses TODO_AUDIT P1)
pub use state_dependent::StateDependentConstants;

// For comprehensive access to all constants, use the submodule namespaces:
// - `acoustic_parameters::*` - Acoustic pressure and intensity references
// - `cavitation::*` - Cavitation thresholds and bubble constants
// - `chemistry::*` - Chemical kinetics and ROS parameters
// - `hounsfield::*` - Hounsfield unit mappings for CT data
// - `optical::*` - Optical wavelengths and refractive indices
// - `state_dependent::*` - Temperature/pressure-dependent constants (TODO_AUDIT P1)
