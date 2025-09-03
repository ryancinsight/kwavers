//! Safety limiting for power modulation

use super::constants::{MAX_AMPLITUDE_RATE, MECHANICAL_INDEX_LIMIT};

/// Safety limiter for preventing excessive power output
#[derive(Debug, Clone)]
pub struct SafetyLimiter {
    max_amplitude: f64,
    max_rate: f64,
    last_output: f64,
    mechanical_index_limit: f64,
}

impl SafetyLimiter {
    /// Create new safety limiter
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_amplitude: 1.0,
            max_rate: MAX_AMPLITUDE_RATE,
            last_output: 0.0,
            mechanical_index_limit: MECHANICAL_INDEX_LIMIT,
        }
    }

    /// Apply safety limiting to amplitude
    pub fn limit(&mut self, amplitude: f64) -> f64 {
        // Clamp to maximum amplitude
        let clamped = amplitude.clamp(0.0, self.max_amplitude);

        // Apply rate limiting (assuming 1ms timestep if not specified)
        // For instantaneous changes (like after reset), allow full change
        let max_change = if self.last_output == 0.0 && amplitude > 0.0 {
            // Allow initial ramp-up without rate limiting
            amplitude
        } else {
            self.max_rate * 0.001 // 1ms timestep default
        };

        let limited = if (clamped - self.last_output).abs() > max_change {
            if clamped > self.last_output {
                self.last_output + max_change
            } else {
                self.last_output - max_change
            }
        } else {
            clamped
        };

        self.last_output = limited;
        limited
    }

    /// Check mechanical index safety
    #[must_use]
    pub fn check_mechanical_index(&self, pressure_mpa: f64, frequency_mhz: f64) -> bool {
        let mi = pressure_mpa / frequency_mhz.sqrt();
        mi <= self.mechanical_index_limit
    }

    /// Set maximum amplitude limit
    pub fn set_max_amplitude(&mut self, max_amplitude: f64) {
        self.max_amplitude = max_amplitude.clamp(0.0, 1.0);
    }

    /// Set maximum rate of change
    pub fn set_max_rate(&mut self, max_rate: f64) {
        self.max_rate = max_rate.abs();
    }

    /// Reset limiter state
    pub fn reset(&mut self) {
        self.last_output = 0.0;
    }
}

impl Default for SafetyLimiter {
    fn default() -> Self {
        Self::new()
    }
}
