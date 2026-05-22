//! Physical constants for acoustic simulations
//!
//! This module provides a single source of truth (SSOT) for all physical constants
//! used throughout the simulation, eliminating magic numbers and ensuring consistency.

pub mod acoustic_parameters;
pub mod cavitation;
pub mod chemistry;
pub mod fundamental;
pub mod hounsfield;
pub mod implants;
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
    ACOUSTIC_ABSORPTION_TISSUE, ATMOSPHERIC_PRESSURE, AVOGADRO, BOLTZMANN, BOND_TRANSFORM_FACTOR,
    B_OVER_A_BLOOD, B_OVER_A_BRAIN, B_OVER_A_BREAST_GLAND, B_OVER_A_CSF, B_OVER_A_FAT,
    B_OVER_A_KIDNEY, B_OVER_A_LIVER, B_OVER_A_LUNG, B_OVER_A_MUSCLE, B_OVER_A_SOFT_TISSUE,
    B_OVER_A_WATER, B_OVER_A_WATER_37C,
    DENSITY_AIR, DENSITY_BLOOD, DENSITY_BRAIN, DENSITY_BREAST_FAT, DENSITY_FAT,
    DENSITY_KIDNEY, DENSITY_LIVER, DENSITY_MUSCLE, DENSITY_TISSUE, DENSITY_WATER_37C,
    DENSITY_WATER_NOMINAL, GAS_CONSTANT, GRAVITY,
    LAME_TO_STIFFNESS_FACTOR, OPTICAL_ABSORPTION_TISSUE_NIR, PLANCK, REDUCED_SCATTERING_TISSUE_NIR,
    SOUND_SPEED_AIR, SOUND_SPEED_BLOOD, SOUND_SPEED_BRAIN, SOUND_SPEED_FAT, SOUND_SPEED_KIDNEY,
    SOUND_SPEED_LIVER, SOUND_SPEED_MUSCLE, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER,
    SOUND_SPEED_WATER_37C, SOUND_SPEED_WATER_SIM, SPEED_OF_LIGHT, SYMMETRY_TOLERANCE,
};

// Water properties (most common medium)
pub use cavitation::{SURFACE_TENSION_TISSUE, SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER};
pub use fundamental::{BULK_MODULUS_WATER, C_WATER, DENSITY_WATER};

// Cavitation control limits
pub use cavitation::{MAX_DUTY_CYCLE, MIN_DUTY_CYCLE};

// Acoustic parameters (most commonly used)
pub use acoustic_parameters::{
    AIR_POLYTROPIC_INDEX, BLOOD_VISCOSITY_37C, DB_TO_NP, DENSITY_SKULL, DIAGNOSTIC_FREQ_MAX,
    DIAGNOSTIC_FREQ_MIN, NP_TO_DB, REFERENCE_FREQUENCY_MHZ, SAMPLING_FREQUENCY_DEFAULT,
    WATER_ABSORPTION_ALPHA_0, WATER_ABSORPTION_POWER, WATER_NONLINEARITY_B_A, WATER_SPECIFIC_HEAT,
    WATER_SURFACE_TENSION_20C, WATER_THERMAL_CONDUCTIVITY, WATER_VAPOR_PRESSURE_20C,
    WATER_VISCOSITY_20C,
};

// Thermodynamic constants
pub use thermodynamic::{
    BODY_TEMPERATURE_C, BODY_TEMPERATURE_K, DC_DT_SOFT_TISSUE, DRHO_DT_SOFT_TISSUE,
    GRUNEISEN_SOFT_TISSUE, GRUNEISEN_WATER_20C, GRUNEISEN_WATER_37C,
    H_VAP_WATER_100C, KELVIN_OFFSET_C, M_WATER, P_CRITICAL_WATER,
    P_TRIPLE_WATER, RHO_C_SOFT_TISSUE, ROOM_TEMPERATURE_C, ROOM_TEMPERATURE_K,
    SPECIFIC_HEAT_BLOOD_PLASMA, SPECIFIC_HEAT_BRAIN_GRAY, SPECIFIC_HEAT_BRAIN_WHITE,
    SPECIFIC_HEAT_CSF, SPECIFIC_HEAT_MINERAL_OIL, SPECIFIC_HEAT_ULTRASOUND_GEL, SPECIFIC_HEAT_URINE,
    SPECIFIC_HEAT_WATER, SPECIFIC_HEAT_WATER_37C,
    THERMAL_CONDUCTIVITY_BRAIN, THERMAL_CONDUCTIVITY_BRAIN_GRAY, THERMAL_CONDUCTIVITY_BLOOD,
    THERMAL_CONDUCTIVITY_CSF, THERMAL_CONDUCTIVITY_FAT, THERMAL_CONDUCTIVITY_KIDNEY,
    THERMAL_CONDUCTIVITY_LIVER, THERMAL_CONDUCTIVITY_MINERAL_OIL, THERMAL_CONDUCTIVITY_MUSCLE,
    THERMAL_CONDUCTIVITY_SKULL, THERMAL_CONDUCTIVITY_ULTRASOUND_GEL, THERMAL_CONDUCTIVITY_URINE,
    THERMAL_CONDUCTIVITY_WATER, THERMAL_CONDUCTIVITY_WATER_37C,
    THERMAL_DIFFUSIVITY_BLOOD, THERMAL_DIFFUSIVITY_TISSUE, THERMAL_DIFFUSIVITY_WATER,
    THERMAL_EXPANSION_SOFT_TISSUE,
    T_BOILING_WATER, T_CRITICAL_WATER, T_TRIPLE_WATER,
};

// Medical constants (FDA limits)
pub use medical::{
    FDA_DERATING_FACTOR, ISPPA_LIMIT, ISPTA_LIMIT, THERMAL_DOSE_THRESHOLD,
    MI_CAVITATION_BOWEL, MI_CAVITATION_BRAIN, MI_CAVITATION_FETAL, MI_CAVITATION_LUNG,
    MI_CAVITATION_OPHTHALMIC, MI_CAVITATION_SOFT_TISSUE,
    MI_LIMIT_BOWEL, MI_LIMIT_BRAIN, MI_LIMIT_FETAL, MI_LIMIT_LUNG, MI_LIMIT_OPHTHALMIC,
    MI_LIMIT_SOFT_TISSUE,
};

// Numerical constants
pub use numerical::{
    ABSORPTION_SINGULARITY_THRESHOLD, CFL_DEFAULT, CFL_FACTOR_3D_FDTD, CFL_SAFETY_FACTOR, EPSILON,
    SMALL_VALUE, SOLVER_TOLERANCE,
};

// State-dependent constants (temperature/pressure-dependent physical properties)
pub use state_dependent::StateDependentConstants;

// For comprehensive access to all constants, use the submodule namespaces:
// - `acoustic_parameters::*` - Acoustic pressure and intensity references
// - `cavitation::*` - Cavitation thresholds and bubble constants
// - `chemistry::*` - Chemical kinetics and ROS parameters
// - `hounsfield::*` - Hounsfield unit mappings for CT data
// - `optical::*` - Optical wavelengths and refractive indices
// - `state_dependent::*` - Temperature/pressure-dependent constants
