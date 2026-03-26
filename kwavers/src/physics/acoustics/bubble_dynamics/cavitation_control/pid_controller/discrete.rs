use super::core::{PIDGains, DEFAULT_OUTPUT_MAX, DEFAULT_OUTPUT_MIN};

/// PID controller using Tustin's method for discrete-time implementation
/// Suitable for digital control systems
#[derive(Debug)]
pub struct TustinPIDController {
    gains: PIDGains,
    sample_time: f64,
    // State variables for discrete implementation
    proportional_state: f64,
    integral_state: f64,
    derivative_state: f64,
    previous_input: f64,
    output_limits: (f64, f64),
}

impl TustinPIDController {
    #[must_use]
    pub fn new(gains: PIDGains, sample_time: f64) -> Self {
        Self {
            gains,
            sample_time,
            proportional_state: 0.0,
            integral_state: 0.0,
            derivative_state: 0.0,
            previous_input: 0.0,
            output_limits: (DEFAULT_OUTPUT_MIN, DEFAULT_OUTPUT_MAX),
        }
    }

    /// Update using discrete-time equations
    pub fn update(&mut self, setpoint: f64, measurement: f64) -> f64 {
        let error = setpoint - measurement;

        // Discrete proportional
        self.proportional_state = self.gains.kp * error;

        // Discrete integral (trapezoidal rule)
        self.integral_state += self.gains.ki * self.sample_time * error;

        // Discrete derivative (backward difference)
        let derivative_raw = (measurement - self.previous_input) / self.sample_time;
        self.derivative_state = self.gains.kd * derivative_raw;

        // Total output
        let output = self.proportional_state + self.integral_state - self.derivative_state;

        // Apply limits and anti-windup
        let limited_output = output.clamp(self.output_limits.0, self.output_limits.1);

        if (output - limited_output).abs() > 1e-6 {
            // Back-calculation anti-windup
            self.integral_state = limited_output - self.proportional_state + self.derivative_state;
        }

        self.previous_input = measurement;

        limited_output
    }

    pub fn reset(&mut self) {
        self.proportional_state = 0.0;
        self.integral_state = 0.0;
        self.derivative_state = 0.0;
        self.previous_input = 0.0;
    }
}
