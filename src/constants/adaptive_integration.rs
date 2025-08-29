//! Adaptive integration constants

/// Safety factor for step size adjustment
pub const SAFETY_FACTOR: f64 = 0.9;

/// Minimum time step [s]
pub const MIN_TIME_STEP: f64 = 1e-15;

/// Maximum time step [s]
pub const MAX_TIME_STEP: f64 = 1e-6;

/// Minimum temperature [K]
pub const MIN_TEMPERATURE: f64 = 273.15;

/// Maximum temperature [K]
pub const MAX_TEMPERATURE: f64 = 1000.0;

/// Minimum radius safety factor
pub const MIN_RADIUS_SAFETY_FACTOR: f64 = 0.01;

/// Maximum velocity fraction
pub const MAX_VELOCITY_FRACTION: f64 = 0.1;

/// Default absolute tolerance
pub const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-12;

/// Default relative tolerance
pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-9;

/// Error control exponent
pub const ERROR_CONTROL_EXPONENT: f64 = 0.2;

/// Half step factor
pub const HALF_STEP_FACTOR: f64 = 0.5;

/// Initial time step fraction
pub const INITIAL_TIME_STEP_FRACTION: f64 = 1e-6;

/// Maximum radius safety factor
pub const MAX_RADIUS_SAFETY_FACTOR: f64 = 100.0;

/// Maximum substeps
pub const MAX_SUBSTEPS: usize = 1000;

/// Maximum time step decrease
pub const MAX_TIME_STEP_DECREASE: f64 = 0.1;

/// Maximum time step increase
pub const MAX_TIME_STEP_INCREASE: f64 = 5.0;
