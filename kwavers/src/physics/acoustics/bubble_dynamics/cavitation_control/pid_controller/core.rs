use std::fmt::Debug;

/// Default proportional gain
pub const DEFAULT_KP: f64 = 1.0;

/// Default integral gain
pub const DEFAULT_KI: f64 = 0.1;

/// Default derivative gain
pub const DEFAULT_KD: f64 = 0.01;

/// Default derivative filter coefficient (0-1, higher = more filtering)
pub const DEFAULT_DERIVATIVE_FILTER: f64 = 0.1;

/// Maximum integral accumulation to prevent windup
pub const DEFAULT_INTEGRAL_LIMIT: f64 = 100.0;

/// Default output saturation limits
pub const DEFAULT_OUTPUT_MIN: f64 = 0.0;
pub const DEFAULT_OUTPUT_MAX: f64 = 1.0;

/// Number of samples for derivative filtering
pub const DERIVATIVE_WINDOW_SIZE: usize = 5;

/// PID controller gains
#[derive(Debug, Clone, Copy)]
pub struct PIDGains {
    pub kp: f64, // Proportional gain
    pub ki: f64, // Integral gain
    pub kd: f64, // Derivative gain
}

impl Default for PIDGains {
    fn default() -> Self {
        Self {
            kp: DEFAULT_KP,
            ki: DEFAULT_KI,
            kd: DEFAULT_KD,
        }
    }
}

/// PID controller configuration
#[derive(Debug, Clone)]
pub struct PIDConfig {
    pub gains: PIDGains,
    pub sample_time: f64,
    pub output_min: f64,
    pub output_max: f64,
    pub integral_limit: f64,
    pub derivative_filter: f64,
    pub setpoint_weighting: f64, // For 2-DOF control (0-1)
}

impl Default for PIDConfig {
    fn default() -> Self {
        Self {
            gains: PIDGains::default(),
            sample_time: 0.001, // 1 ms default
            output_min: DEFAULT_OUTPUT_MIN,
            output_max: DEFAULT_OUTPUT_MAX,
            integral_limit: DEFAULT_INTEGRAL_LIMIT,
            derivative_filter: DEFAULT_DERIVATIVE_FILTER,
            setpoint_weighting: 1.0, // Standard PID
        }
    }
}

/// Error integral state
#[derive(Debug, Clone)]
pub struct ErrorIntegral {
    pub value: f64,
    pub limit: f64,
}

impl ErrorIntegral {
    #[must_use] 
    pub fn new(limit: f64) -> Self {
        Self { value: 0.0, limit }
    }

    pub fn update(&mut self, error: f64, dt: f64) {
        // Update integral with proper clamping
        let new_value = error.mul_add(dt, self.value);
        // Anti-windup: clamp integral to prevent windup
        self.value = new_value.clamp(-self.limit, self.limit);
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }
}

/// Controller output with diagnostics
#[derive(Debug, Clone)]
pub struct ControllerOutput {
    pub control_signal: f64,
    pub proportional_term: f64,
    pub integral_term: f64,
    pub derivative_term: f64,
    pub error: f64,
    pub saturated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default PIDConfig satisfies output ordering invariant.
    #[test]
    fn default_pid_config_output_ordering() {
        let cfg = PIDConfig::default();
        assert!(cfg.output_min < cfg.output_max,
            "output_min ({}) must be < output_max ({})", cfg.output_min, cfg.output_max);
        assert!(cfg.sample_time > 0.0, "sample_time must be positive");
        assert!(cfg.integral_limit > 0.0, "integral_limit must be positive");
    }

    /// PIDGains::default matches canonical constants.
    #[test]
    fn default_pid_gains_match_constants() {
        let g = PIDGains::default();
        assert!((g.kp - DEFAULT_KP).abs() < 1e-15);
        assert!((g.ki - DEFAULT_KI).abs() < 1e-15);
        assert!((g.kd - DEFAULT_KD).abs() < 1e-15);
    }

    /// ErrorIntegral::update accumulates correctly and clamps at limit.
    ///
    /// With limit=1.0, dt=1.0, error=2.0: after one update, value = clamp(0+2·1, -1, 1) = 1.0.
    #[test]
    fn error_integral_update_clamps_at_limit() {
        let mut ei = ErrorIntegral::new(1.0);
        ei.update(2.0, 1.0); // error * dt = 2 > limit
        assert!((ei.value - 1.0).abs() < 1e-15,
            "integral must be clamped at limit=1.0, got {}", ei.value);
    }

    /// ErrorIntegral::reset sets value to zero.
    #[test]
    fn error_integral_reset_zeros_value() {
        let mut ei = ErrorIntegral::new(10.0);
        ei.update(1.0, 1.0); // value = 1.0
        ei.reset();
        assert!((ei.value).abs() < 1e-15, "reset must zero the integral");
    }

    /// ErrorIntegral accumulates correctly below the limit.
    ///
    /// With limit=10.0, error=0.5, dt=2.0 → accumulated = 0.5·2.0 = 1.0 < limit.
    #[test]
    fn error_integral_accumulates_below_limit() {
        let mut ei = ErrorIntegral::new(10.0);
        ei.update(0.5, 2.0);
        assert!((ei.value - 1.0).abs() < 1e-15,
            "integral must be 1.0 (got {})", ei.value);
    }
}
