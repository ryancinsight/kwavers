//! Cavitation-related constants

/// Blake threshold pressure ratio for inertial cavitation
pub const BLAKE_THRESHOLD: f64 = 0.541;

/// Typical initial bubble radius (m)
pub const INITIAL_BUBBLE_RADIUS: f64 = 5e-6;

/// Surface tension of water at 20°C (N/m)
pub const SURFACE_TENSION_WATER: f64 = 0.0728;

/// Viscosity of water at 20°C (Pa·s)
pub const VISCOSITY_WATER: f64 = 1.002e-3;

/// Polytrophic exponent for air
pub const POLYTROPIC_EXPONENT_AIR: f64 = 1.4;

/// Van der Waals hard core radius for bubble (m)
pub const VAN_DER_WAALS_RADIUS: f64 = 8.86e-10;

/// Mechanical index threshold for bioeffects
pub const MECHANICAL_INDEX_THRESHOLD: f64 = 0.7;

/// Thermal index threshold for bioeffects
pub const THERMAL_INDEX_THRESHOLD: f64 = 6.0;

/// Cavitation inception threshold (MPa)
pub const CAVITATION_THRESHOLD_WATER: f64 = -30.0;

/// Typical bubble damping constant
pub const BUBBLE_DAMPING_CONSTANT: f64 = 1.5e-9;