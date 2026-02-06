//! Cavitation-related constants

/// Blake threshold pressure ratio for inertial cavitation
pub const BLAKE_THRESHOLD: f64 = 0.541;

/// Typical initial bubble radius (m)
pub const INITIAL_BUBBLE_RADIUS: f64 = 5e-6;

/// Surface tension of water at 20°C (N/m)
pub const SURFACE_TENSION_WATER: f64 = 0.0728;

/// Viscosity of water at 20°C (Pa·s)
pub const VISCOSITY_WATER: f64 = 1.002e-3;

/// Vapor pressure of water at 20°C (Pa)
pub const VAPOR_PRESSURE_WATER: f64 = 2339.0;

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

// ============================================================================
// Cavitation Damage Parameters
// ============================================================================

/// Compression factor exponent for damage calculation
pub const COMPRESSION_FACTOR_EXPONENT: f64 = 2.0;

/// Default bubble concentration factor
pub const DEFAULT_CONCENTRATION_FACTOR: f64 = 1e5;

/// Default fatigue rate for material
pub const DEFAULT_FATIGUE_RATE: f64 = 0.01;

/// Default pit formation efficiency
pub const DEFAULT_PIT_EFFICIENCY: f64 = 0.1;

/// Default cavitation threshold pressure (Pa)
pub const DEFAULT_THRESHOLD_PRESSURE: f64 = -1e5;

/// Impact energy coefficient
pub const IMPACT_ENERGY_COEFFICIENT: f64 = 0.5;

/// Material removal efficiency factor
pub const MATERIAL_REMOVAL_EFFICIENCY: f64 = 0.05;

// ============================================================================
// Bubble Dynamics Limits
// ============================================================================

/// Maximum bubble radius (m)
pub const MAX_RADIUS: f64 = 1e-3;

/// Minimum bubble radius (m)
pub const MIN_RADIUS: f64 = 1e-9;

/// Conversion factor from bar·L² to Pa·m⁶
pub const BAR_L2_TO_PA_M6: f64 = 1e-7;

/// Conversion factor from liters to cubic meters
pub const L_TO_M3: f64 = 1e-3;

/// Minimum Peclet number for thermal effects
pub const MIN_PECLET_NUMBER: f64 = 10.0;

/// Peclet number scaling factor
pub const PECLET_SCALING_FACTOR: f64 = 0.1;

/// Surface tension coefficient multiplier
pub const SURFACE_TENSION_COEFF: f64 = 2.0;

// ============================================================================
// Power Modulation Limits
// ============================================================================

/// Minimum duty cycle for power modulation
pub const MIN_DUTY_CYCLE: f64 = 0.01;

/// Maximum duty cycle for power modulation
pub const MAX_DUTY_CYCLE: f64 = 1.0;
