//! Constants for power modulation

#![allow(dead_code)] // Configuration constants for power modulation functionality

/// Default pulse repetition frequency (PRF) in Hz
pub const DEFAULT_PRF: f64 = 100.0;

/// Default duty cycle (0-1)
pub const DEFAULT_DUTY_CYCLE: f64 = 0.5;

/// Minimum duty cycle to prevent complete shutdown
pub const MIN_DUTY_CYCLE: f64 = 0.01;

/// Maximum duty cycle for safety
pub const MAX_DUTY_CYCLE: f64 = 0.95;

/// Default ramp time for smooth transitions (seconds)
pub const DEFAULT_RAMP_TIME: f64 = 0.01;

/// Maximum amplitude change rate (per second)
pub const MAX_AMPLITUDE_RATE: f64 = 10.0;

/// Safety threshold for mechanical index
pub const MECHANICAL_INDEX_LIMIT: f64 = 1.9;

/// Default filter time constant (seconds)
pub const DEFAULT_FILTER_TIME_CONSTANT: f64 = 0.1;

/// Maximum allowed pressure (Pa)
pub const MAX_PRESSURE_PA: f64 = 10e6;

/// Minimum modulation frequency (Hz)
pub const MIN_MODULATION_FREQ: f64 = 0.1;

/// Maximum modulation frequency (Hz)
pub const MAX_MODULATION_FREQ: f64 = 1000.0;
