//! PID Controller for Cavitation Feedback Control
//! 
//! Implements a discrete-time PID controller with anti-windup and derivative filtering.
//! 
//! References:
//! - Åström & Hägglund (2006): "Advanced PID Control"
//! - Franklin et al. (2015): "Feedback Control of Dynamic Systems"

use std::collections::VecDeque;
use std::fmt::Debug;

// PID controller constants
/// Default proportional gain
const DEFAULT_KP: f64 = 1.0;

/// Default integral gain
const DEFAULT_KI: f64 = 0.1;

/// Default derivative gain
const DEFAULT_KD: f64 = 0.01;

/// Default derivative filter coefficient (0-1, higher = more filtering)
const DEFAULT_DERIVATIVE_FILTER: f64 = 0.1;

/// Maximum integral accumulation to prevent windup
const DEFAULT_INTEGRAL_LIMIT: f64 = 100.0;

/// Default output saturation limits
const DEFAULT_OUTPUT_MIN: f64 = 0.0;
const DEFAULT_OUTPUT_MAX: f64 = 1.0;

/// Number of samples for derivative filtering
const DERIVATIVE_WINDOW_SIZE: usize = 5;

/// PID controller gains
#[derive(Debug, Clone, Copy)]
pub struct PIDGains {
    pub kp: f64,  // Proportional gain
    pub ki: f64,  // Integral gain
    pub kd: f64,  // Derivative gain
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
    pub setpoint_weighting: f64,  // For 2-DOF control (0-1)
}

impl Default for PIDConfig {
    fn default() -> Self {
        Self {
            gains: PIDGains::default(),
            sample_time: 0.001,  // 1 ms default
            output_min: DEFAULT_OUTPUT_MIN,
            output_max: DEFAULT_OUTPUT_MAX,
            integral_limit: DEFAULT_INTEGRAL_LIMIT,
            derivative_filter: DEFAULT_DERIVATIVE_FILTER,
            setpoint_weighting: 1.0,  // Standard PID
        }
    }
}

/// Error integral state
#[derive(Debug, Clone)]
pub struct ErrorIntegral {
    value: f64,
    limit: f64,
}

impl ErrorIntegral {
    fn new(limit: f64) -> Self {
        Self { value: 0.0, limit }
    }
    
    fn update(&mut self, error: f64, dt: f64) {
        self.value += error * dt;
        // Anti-windup: clamp integral
        self.value = self.value.clamp(-self.limit, self.limit);
    }
    
    fn reset(&mut self) {
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

/// PID Controller with anti-windup and derivative filtering
#[derive(Debug)]
pub struct PIDController {
    config: PIDConfig,
    integral: ErrorIntegral,
    previous_error: f64,
    derivative_filter: VecDeque<f64>,
    setpoint: f64,
    last_output: f64,
    initialized: bool,
}

impl PIDController {
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
        let p_term = self.config.gains.kp * 
            (self.config.setpoint_weighting * self.setpoint - measurement);
        
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
        let saturated = control_signal < self.config.output_min || 
                       control_signal > self.config.output_max;
        
        // Apply output limits
        control_signal = control_signal.clamp(
            self.config.output_min,
            self.config.output_max
        );
        
        // Back-calculation anti-windup
        if saturated {
            let excess = control_signal - (p_term + i_term + d_term);
            self.integral.value -= excess * self.config.sample_time / self.config.gains.ki;
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
    pub fn config(&self) -> &PIDConfig {
        &self.config
    }
    
    /// Get current integral value (for monitoring)
    pub fn integral_value(&self) -> f64 {
        self.integral.value
    }
}

/// Discrete-time PID controller using Tustin's method
/// Better for digital implementation
pub struct DiscretePIDController {
    gains: PIDGains,
    sample_time: f64,
    // State variables for discrete implementation
    proportional_state: f64,
    integral_state: f64,
    derivative_state: f64,
    previous_input: f64,
    output_limits: (f64, f64),
}

impl DiscretePIDController {
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pid_step_response() {
        let config = PIDConfig {
            gains: PIDGains { kp: 2.0, ki: 1.0, kd: 0.5 },
            sample_time: 0.01,
            ..Default::default()
        };
        
        let mut controller = PIDController::new(config);
        controller.set_setpoint(1.0);
        
        let mut measurement = 0.0;
        for _ in 0..100 {
            let output = controller.update(measurement);
            // Simple first-order system simulation
            measurement += output.control_signal * 0.01;
            measurement *= 0.99; // Small damping
        }
        
        // Should converge close to setpoint
        assert!((measurement - 1.0).abs() < 0.1);
    }
    
    #[test]
    fn test_anti_windup() {
        let config = PIDConfig {
            gains: PIDGains { kp: 1.0, ki: 10.0, kd: 0.0 },
            integral_limit: 1.0,
            ..Default::default()
        };
        
        let mut controller = PIDController::new(config);
        controller.set_setpoint(10.0); // Large setpoint
        
        // Run with measurement stuck at 0
        for _ in 0..100 {
            controller.update(0.0);
        }
        
        // Integral should be limited
        assert!(controller.integral_value().abs() <= 1.0);
    }
}