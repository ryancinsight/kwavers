//! Physical constants for Mie scattering calculations

/// Speed of light in vacuum [m/s]
pub const C: f64 = 2.99792458e8;
/// Vacuum permittivity [F/m]
pub const EPSILON_0: f64 = crate::core::constants::fundamental::VACUUM_PERMITTIVITY;
/// Vacuum permeability [H/m]
pub const MU_0: f64 = 4.0 * std::f64::consts::PI * 1e-7;
