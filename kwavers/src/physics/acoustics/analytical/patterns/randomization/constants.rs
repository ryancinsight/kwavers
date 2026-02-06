//! Constants for phase randomization

use std::f64::consts::PI;

/// Maximum phase shift for randomization (radians)
pub const MAX_PHASE_SHIFT: f64 = 2.0 * PI;

/// Default number of phase states for discrete randomization
pub const DEFAULT_PHASE_STATES: usize = 4;

/// Minimum switching period for temporal randomization (seconds)
pub const MIN_SWITCHING_PERIOD: f64 = 1e-6;

/// Default correlation length for spatial randomization (meters)
pub const DEFAULT_CORRELATION_LENGTH: f64 = 0.001; // 1mm

/// Phase tolerance for comparison (radians)
pub const PHASE_TOLERANCE: f64 = 1e-10;

/// Default seed for reproducible randomization
pub const DEFAULT_SEED: u64 = 42;
