//! Numerical stability constants

/// Maximum allowed CFL number
pub const MAX_CFL: f64 = 1.0;

/// Minimum allowed CFL number
pub const MIN_CFL: f64 = 0.01;

/// Maximum allowed time step [s]
pub const MAX_DT: f64 = 1e-3;

/// Minimum allowed time step [s]
pub const MIN_DT: f64 = 1e-15;

/// Maximum allowed grid spacing [m]
pub const MAX_DX: f64 = 1.0;

/// Minimum allowed grid spacing [m]
pub const MIN_DX: f64 = 1e-6;

/// Stability safety margin
pub const STABILITY_MARGIN: f64 = 0.95;

/// Maximum allowed gradient
pub const MAX_GRADIENT: f64 = 1e6;

/// Minimum allowed density [kg/m³]
pub const MIN_DENSITY: f64 = 0.1;

/// Maximum allowed density [kg/m³]
pub const MAX_DENSITY: f64 = 10000.0;
