//! Safety monitoring for control system

use super::super::detection::CavitationMetrics;
use super::types::{ControlOutput, SafetyLimits};

const SAFETY_SHUTDOWN_THRESHOLD: f64 = 0.95;
const SAFETY_RECOVERY_THRESHOLD: f64 = 0.7;

/// Safety monitor for control system
#[derive(Debug)]
pub struct SafetyMonitor {
    limits: SafetyLimits,
    emergency_stop: bool,
    violation_count: usize,
}

impl SafetyMonitor {
    #[must_use]
    pub fn new(limits: SafetyLimits) -> Self {
        Self {
            limits,
            emergency_stop: false,
            violation_count: 0,
        }
    }

    pub fn check_safety(&mut self, metrics: &CavitationMetrics) -> bool {
        // Check for excessive cavitation
        if metrics.intensity > SAFETY_SHUTDOWN_THRESHOLD {
            self.emergency_stop = true;
            self.violation_count += 1;
            return false;
        }

        // Check for recovery
        if self.emergency_stop && metrics.intensity < SAFETY_RECOVERY_THRESHOLD {
            self.emergency_stop = false;
        }

        !self.emergency_stop
    }

    #[must_use]
    pub fn apply_safety_limits(&self, mut output: ControlOutput) -> ControlOutput {
        if self.emergency_stop {
            output.amplitude = 0.0;
            output.safety_limited = true;
            output.control_active = false;
        } else if output.cavitation_intensity > self.limits.max_intensity {
            // Scale down amplitude
            let scale = self.limits.max_intensity / output.cavitation_intensity;
            output.amplitude *= scale;
            output.safety_limited = true;
        }

        output
    }

    #[must_use]
    pub fn is_emergency_stop(&self) -> bool {
        self.emergency_stop
    }

    pub fn reset(&mut self) {
        self.emergency_stop = false;
        self.violation_count = 0;
    }

    #[must_use]
    pub fn get_violation_count(&self) -> usize {
        self.violation_count
    }
}
