//! Constants for power modulation

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
pub const MECHANICAL_INDEX_LIMIT: f64 = kwavers_core::constants::medical::MI_LIMIT_SOFT_TISSUE;

/// Default filter time constant (seconds)
pub const DEFAULT_FILTER_TIME_CONSTANT: f64 = 0.1;

const _: () = {
    assert!(MIN_DUTY_CYCLE < DEFAULT_DUTY_CYCLE);
    assert!(DEFAULT_DUTY_CYCLE < MAX_DUTY_CYCLE);
    assert!(MAX_DUTY_CYCLE < 1.0);
    assert!(MECHANICAL_INDEX_LIMIT > 0.0);
    assert!(MAX_AMPLITUDE_RATE > 0.0);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Duty cycle constants satisfy: MIN < DEFAULT < MAX, all in (0, 1).
    #[test]
    fn duty_cycle_ordering() {
        let min_duty_cycle = MIN_DUTY_CYCLE;
        let default_duty_cycle = DEFAULT_DUTY_CYCLE;
        let max_duty_cycle = MAX_DUTY_CYCLE;
        assert!(min_duty_cycle < default_duty_cycle);
        assert!(default_duty_cycle < max_duty_cycle);
        assert!(max_duty_cycle < 1.0);
    }

    /// Safety-critical constants are positive.
    #[test]
    fn safety_constants_positive() {
        let mechanical_index_limit = MECHANICAL_INDEX_LIMIT;
        let max_amplitude_rate = MAX_AMPLITUDE_RATE;
        assert!(mechanical_index_limit.is_sign_positive());
        assert!(max_amplitude_rate.is_sign_positive());
    }
}
