use std::collections::VecDeque;

use super::core::{ControllerOutput, ErrorIntegral, PIDConfig, PIDGains, DERIVATIVE_WINDOW_SIZE};

/// PID Controller with anti-windup and derivative filtering
#[derive(Debug)]
pub struct PIDController {
    config: PIDConfig,
    integral: ErrorIntegral,
    previous_error: f64,
    derivative_filter: VecDeque<f64>,
    setpoint: f64,
    #[allow(dead_code)]
    last_output: f64,
    initialized: bool,
}

impl PIDController {
    #[must_use]
    pub fn new(config: PIDConfig) -> Self {
        Self {
            integral: ErrorIntegral::new(config.integral_limit),
            config,
            previous_error: 0.0,
            derivative_filter: VecDeque::with_capacity(DERIVATIVE_WINDOW_SIZE),
            setpoint: 0.0,
            last_output: 0.0,
            initialized: false,
        }
    }

    /// Set the target setpoint
    pub fn set_setpoint(&mut self, setpoint: f64) {
        self.setpoint = setpoint;
    }

    /// Reset the controller state
    pub fn reset(&mut self) {
        self.integral.reset();
        self.previous_error = 0.0;
        self.derivative_filter.clear();
        self.last_output = 0.0;
        self.initialized = false;
    }

    /// Update gains dynamically
    pub fn update_gains(&mut self, gains: PIDGains) {
        self.config.gains = gains;
    }

    /// Compute control output using velocity form to reduce bumps
    pub fn update(&mut self, measurement: f64) -> ControllerOutput {
        let error = self.setpoint - measurement;

        // Initialize on first call
        if !self.initialized {
            self.previous_error = error;
            self.initialized = true;
        }

        // Proportional term with setpoint weighting
        let p_term =
            self.config.gains.kp * (self.config.setpoint_weighting * self.setpoint - measurement);

        // Integral term with anti-windup
        self.integral.update(error, self.config.sample_time);
        let i_term = self.config.gains.ki * self.integral.value;

        // Derivative term with filtering
        let raw_derivative = (error - self.previous_error) / self.config.sample_time;
        let filtered_derivative = self.filter_derivative(raw_derivative);
        let d_term = self.config.gains.kd * filtered_derivative;

        // Calculate total control signal
        let mut control_signal = p_term + i_term + d_term;

        // Check for saturation
        let saturated =
            control_signal < self.config.output_min || control_signal > self.config.output_max;

        // Apply output limits
        control_signal = control_signal.clamp(self.config.output_min, self.config.output_max);

        // Back-calculation anti-windup
        if saturated && self.config.gains.ki != 0.0 {
            let excess = control_signal - (p_term + i_term + d_term);
            let adjustment = excess * self.config.sample_time / self.config.gains.ki;
            // Apply clamping after adjustment
            self.integral.value =
                (self.integral.value - adjustment).clamp(-self.integral.limit, self.integral.limit);
        }

        // Update state for next iteration
        self.previous_error = error;
        self.last_output = control_signal;

        ControllerOutput {
            control_signal,
            proportional_term: p_term,
            integral_term: i_term,
            derivative_term: d_term,
            error,
            saturated,
        }
    }

    /// Filter derivative to reduce noise
    fn filter_derivative(&mut self, raw_derivative: f64) -> f64 {
        self.derivative_filter.push_back(raw_derivative);

        if self.derivative_filter.len() > DERIVATIVE_WINDOW_SIZE {
            self.derivative_filter.pop_front();
        }

        if self.derivative_filter.is_empty() {
            return raw_derivative;
        }

        // Moving average filter
        let sum: f64 = self.derivative_filter.iter().sum();
        sum / self.derivative_filter.len() as f64
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &PIDConfig {
        &self.config
    }

    /// Get current integral value (for monitoring)
    #[must_use]
    pub fn integral_value(&self) -> f64 {
        self.integral.value
    }
}
