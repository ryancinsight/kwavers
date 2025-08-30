//! Amplitude control for power modulation

use super::constants::*;
use super::filters::ExponentialFilter;

/// Amplitude controller with feedback
#[derive(Debug, Clone)]
pub struct AmplitudeController {
    target_amplitude: f64,
    current_amplitude: f64,
    ramp_rate: f64,
    filter: ExponentialFilter,
}

impl AmplitudeController {
    /// Create new amplitude controller
    pub fn new(initial_amplitude: f64) -> Self {
        Self {
            target_amplitude: initial_amplitude,
            current_amplitude: initial_amplitude,
            ramp_rate: MAX_AMPLITUDE_RATE,
            filter: ExponentialFilter::new(DEFAULT_FILTER_TIME_CONSTANT),
        }
    }

    /// Set target amplitude
    pub fn set_target(&mut self, target: f64) {
        self.target_amplitude = target.clamp(0.0, 1.0);
    }

    /// Update amplitude with ramping
    pub fn update(&mut self, dt: f64) -> f64 {
        let max_change = self.ramp_rate * dt;
        let diff = self.target_amplitude - self.current_amplitude;
        
        if diff.abs() <= max_change {
            self.current_amplitude = self.target_amplitude;
        } else {
            self.current_amplitude += diff.signum() * max_change;
        }
        
        // Apply filtering for smooth transitions
        self.filter.filter(self.current_amplitude)
    }

    /// Get current amplitude
    pub fn get_amplitude(&self) -> f64 {
        self.current_amplitude
    }

    /// Set ramp rate
    pub fn set_ramp_rate(&mut self, rate: f64) {
        self.ramp_rate = rate.abs();
    }

    /// Reset controller
    pub fn reset(&mut self) {
        self.current_amplitude = 0.0;
        self.filter.reset();
    }
}