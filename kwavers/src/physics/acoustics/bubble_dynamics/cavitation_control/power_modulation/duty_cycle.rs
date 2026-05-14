//! Duty cycle control for power modulation

use super::constants::{MAX_DUTY_CYCLE, MIN_DUTY_CYCLE};

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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn average_power(&self, peak_power: f64) -> f64 {
        peak_power * self.current_duty_cycle
    }
}

#[cfg(test)]
mod tests {
    use super::super::constants::{MAX_DUTY_CYCLE, MIN_DUTY_CYCLE};
    use super::*;

    /// new(0.5) clamps to [MIN, MAX] and stores 0.5 when 0.5 ∈ [MIN, MAX].
    #[test]
    fn new_stores_initial_duty_cycle_within_bounds() {
        let dc = DutyCycleController::new(0.5);
        assert!(
            (dc.get_duty_cycle() - 0.5).abs() < 1e-15,
            "initial duty cycle must be 0.5; got {}",
            dc.get_duty_cycle()
        );
    }

    /// new with out-of-range value clamps to MIN_DUTY_CYCLE.
    #[test]
    fn new_clamps_below_minimum() {
        let dc = DutyCycleController::new(0.0); // below MIN_DUTY_CYCLE=0.01
        assert!(
            (dc.get_duty_cycle() - MIN_DUTY_CYCLE).abs() < 1e-15,
            "clamped duty cycle must equal MIN_DUTY_CYCLE={MIN_DUTY_CYCLE}; got {}",
            dc.get_duty_cycle()
        );
    }

    /// average_power = peak * current_duty_cycle (linear by definition).
    ///
    /// With initial_duty_cycle=0.5, peak=2.0 → average = 1.0.
    #[test]
    fn average_power_is_linear_in_duty_cycle() {
        let dc = DutyCycleController::new(0.5);
        let avg = dc.average_power(2.0);
        assert!(
            (avg - 1.0).abs() < 1e-14,
            "average_power must be 1.0; got {avg}"
        );
    }

    /// set_limits enforces that current_duty_cycle is clamped to the new bounds.
    #[test]
    fn set_limits_clamps_current_duty_cycle() {
        let mut dc = DutyCycleController::new(MAX_DUTY_CYCLE); // starts at 0.95
        dc.set_limits(MIN_DUTY_CYCLE, 0.6); // new max=0.6
        assert!(
            dc.get_duty_cycle() <= 0.6,
            "duty cycle must be clamped to new max 0.6; got {}",
            dc.get_duty_cycle()
        );
    }
}
