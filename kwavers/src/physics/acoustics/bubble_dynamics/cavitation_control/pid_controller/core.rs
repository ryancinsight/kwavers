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
    pub fn new(limit: f64) -> Self {
        Self { value: 0.0, limit }
    }

    pub fn update(&mut self, error: f64, dt: f64) {
        // Update integral with proper clamping
        let new_value = self.value + error * dt;
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
