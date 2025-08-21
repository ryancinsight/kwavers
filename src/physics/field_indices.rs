//! Unified field indices - Single Source of Truth (SSOT)
//!
//! This module provides all field indices used throughout the simulation.
//! All modules should import from here rather than defining their own.

/// Acoustic pressure field
pub const PRESSURE_IDX: usize = 0;

/// Temperature field
pub const TEMPERATURE_IDX: usize = 1;

/// Light intensity/fluence field
pub const LIGHT_IDX: usize = 2;

/// Velocity X component
pub const VX_IDX: usize = 3;

/// Velocity Y component
pub const VY_IDX: usize = 4;

/// Velocity Z component
pub const VZ_IDX: usize = 5;

/// Stress tensor XX component
pub const STRESS_XX_IDX: usize = 6;

/// Stress tensor YY component
pub const STRESS_YY_IDX: usize = 7;

/// Stress tensor ZZ component
pub const STRESS_ZZ_IDX: usize = 8;

/// Stress tensor XY component
pub const STRESS_XY_IDX: usize = 9;

/// Stress tensor XZ component
pub const STRESS_XZ_IDX: usize = 10;

/// Stress tensor YZ component
pub const STRESS_YZ_IDX: usize = 11;

/// Bubble radius field
pub const BUBBLE_RADIUS_IDX: usize = 12;

/// Bubble velocity field
pub const BUBBLE_VELOCITY_IDX: usize = 13;

/// Chemical concentration field
pub const CHEMICAL_IDX: usize = 14;

/// Density field
pub const DENSITY_IDX: usize = 15;

/// Sound speed field
pub const SOUND_SPEED_IDX: usize = 16;

// Aliases for stress components (for compatibility)
pub const SXX_IDX: usize = STRESS_XX_IDX;
pub const SYY_IDX: usize = STRESS_YY_IDX;
pub const SZZ_IDX: usize = STRESS_ZZ_IDX;
pub const SXY_IDX: usize = STRESS_XY_IDX;
pub const SXZ_IDX: usize = STRESS_XZ_IDX;
pub const SYZ_IDX: usize = STRESS_YZ_IDX;

/// Total number of fields
pub const TOTAL_FIELDS: usize = 17;

/// Get field name from index
pub fn field_name(idx: usize) -> &'static str {
    match idx {
        PRESSURE_IDX => "Pressure",
        TEMPERATURE_IDX => "Temperature",
        LIGHT_IDX => "Light",
        VX_IDX => "Velocity X",
        VY_IDX => "Velocity Y",
        VZ_IDX => "Velocity Z",
        STRESS_XX_IDX => "Stress XX",
        STRESS_YY_IDX => "Stress YY",
        STRESS_ZZ_IDX => "Stress ZZ",
        STRESS_XY_IDX => "Stress XY",
        STRESS_XZ_IDX => "Stress XZ",
        STRESS_YZ_IDX => "Stress YZ",
        BUBBLE_RADIUS_IDX => "Bubble Radius",
        BUBBLE_VELOCITY_IDX => "Bubble Velocity",
        CHEMICAL_IDX => "Chemical Concentration",
        DENSITY_IDX => "Density",
        SOUND_SPEED_IDX => "Sound Speed",
        _ => "Unknown",
    }
}
