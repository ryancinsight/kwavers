//! Duty cycle control for power modulation

use super::constants::*;

/// Duty cycle controller for pulsed operation
#[derive(Debug, Clone)]
pub struct DutyCycleController {
    target_duty_cycle: f64,
    current_duty_cycle: f64,
    adjustment_rate: f64,
    min_duty_cycle: f64,
    max_duty_cycle: f64,
}

impl DutyCycleController {
    /// Create new duty cycle controller
    pub fn new(initial_duty_cycle: f64) -> Self {
        Self {
            target_duty_cycle: initial_duty_cycle.clamp(MIN_DUTY_CYCLE, MAX_DUTY_CYCLE),
            current_duty_cycle: initial_duty_cycle.clamp(MIN_DUTY_CYCLE, MAX_DUTY_CYCLE),
            adjustment_rate: 0.1, // 10% per second
            min_duty_cycle: MIN_DUTY_CYCLE,
            max_duty_cycle: MAX_DUTY_CYCLE,
        }
    }

    /// Set target duty cycle
    pub fn set_target(&mut self, target: f64) {
        self.target_duty_cycle = target.clamp(self.min_duty_cycle, self.max_duty_cycle);
    }

    /// Update duty cycle with rate limiting
    pub fn update(&mut self, dt: f64) -> f64 {
        let max_change = self.adjustment_rate * dt;
        let diff = self.target_duty_cycle - self.current_duty_cycle;

        if diff.abs() <= max_change {
            self.current_duty_cycle = self.target_duty_cycle;
        } else {
            self.current_duty_cycle += diff.signum() * max_change;
        }

        self.current_duty_cycle
    }

    /// Get current duty cycle
    pub fn get_duty_cycle(&self) -> f64 {
        self.current_duty_cycle
    }

    /// Set adjustment rate
    pub fn set_adjustment_rate(&mut self, rate: f64) {
        self.adjustment_rate = rate.abs();
    }

    /// Set duty cycle limits
    pub fn set_limits(&mut self, min: f64, max: f64) {
        self.min_duty_cycle = min.max(MIN_DUTY_CYCLE);
        self.max_duty_cycle = max.min(MAX_DUTY_CYCLE);

        // Ensure current values are within new limits
        self.current_duty_cycle = self
            .current_duty_cycle
            .clamp(self.min_duty_cycle, self.max_duty_cycle);
        self.target_duty_cycle = self
            .target_duty_cycle
            .clamp(self.min_duty_cycle, self.max_duty_cycle);
    }

    /// Calculate average power for current duty cycle
    pub fn average_power(&self, peak_power: f64) -> f64 {
        peak_power * self.current_duty_cycle
    }
}
