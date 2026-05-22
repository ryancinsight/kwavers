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

/// Safety threshold for mechanical index — delegates to `medical::MI_LIMIT_SOFT_TISSUE` (FDA 2019).
pub const MECHANICAL_INDEX_LIMIT: f64 = crate::core::constants::medical::MI_LIMIT_SOFT_TISSUE;

/// Default filter time constant (seconds)
pub const DEFAULT_FILTER_TIME_CONSTANT: f64 = 0.1;

/// Maximum allowed pressure (Pa)
pub const MAX_PRESSURE_PA: f64 = 10e6;

/// Minimum modulation frequency (Hz)
pub const MIN_MODULATION_FREQ: f64 = 0.1;

/// Maximum modulation frequency (Hz)
pub const MAX_MODULATION_FREQ: f64 = 1000.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// Duty cycle constants satisfy: MIN < DEFAULT < MAX, all in (0, 1).
    #[test]
    fn duty_cycle_ordering() {
        assert!(
            MIN_DUTY_CYCLE < DEFAULT_DUTY_CYCLE,
            "MIN_DUTY_CYCLE ({}) must be < DEFAULT_DUTY_CYCLE ({})",
            MIN_DUTY_CYCLE,
            DEFAULT_DUTY_CYCLE
        );
        assert!(
            DEFAULT_DUTY_CYCLE < MAX_DUTY_CYCLE,
            "DEFAULT_DUTY_CYCLE ({}) must be < MAX_DUTY_CYCLE ({})",
            DEFAULT_DUTY_CYCLE,
            MAX_DUTY_CYCLE
        );
        assert!(
            MAX_DUTY_CYCLE < 1.0,
            "MAX_DUTY_CYCLE must be < 1.0 (safety margin)"
        );
    }

    /// Modulation frequency range is valid.
    #[test]
    fn modulation_freq_ordering() {
        assert!(MIN_MODULATION_FREQ > 0.0);
        assert!(
            MIN_MODULATION_FREQ < MAX_MODULATION_FREQ,
            "MIN_MODULATION_FREQ ({}) must be < MAX_MODULATION_FREQ ({})",
            MIN_MODULATION_FREQ,
            MAX_MODULATION_FREQ
        );
    }

    /// Safety-critical constants are positive.
    #[test]
    fn safety_constants_positive() {
        assert!(MECHANICAL_INDEX_LIMIT > 0.0);
        assert!(MAX_PRESSURE_PA > 0.0);
        assert!(MAX_AMPLITUDE_RATE > 0.0);
    }
}
