//! Bubble dynamics constants

/// Conversion from bar·L² to Pa·m⁶
pub const BAR_L2_TO_PA_M6: f64 = 1e-1;

/// Conversion from L to m³
pub const L_TO_M3: f64 = 1e-3;

/// Viscous stress coefficient
pub const VISCOUS_STRESS_COEFF: f64 = 4.0;

/// Surface tension coefficient
pub const SURFACE_TENSION_COEFF: f64 = 2.0;

/// Kinetic energy coefficient
pub const KINETIC_ENERGY_COEFF: f64 = 1.5;

/// Minimum Peclet number
pub const MIN_PECLET_NUMBER: f64 = 1e-3;

/// Peclet scaling factor
pub const PECLET_SCALING_FACTOR: f64 = 1.0;

/// Water latent heat of vaporization [J/kg]
pub const WATER_LATENT_HEAT_VAPORIZATION: f64 = 2.26e6;

/// Maximum bubble radius [m]
pub const MAX_RADIUS: f64 = 1e-2;

/// Minimum bubble radius [m]
pub const MIN_RADIUS: f64 = 1e-9;
