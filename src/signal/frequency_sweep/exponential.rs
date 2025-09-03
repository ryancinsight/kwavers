// frequency_sweep/exponential.rs - Exponential frequency sweep

use super::{
    constants::{EPSILON, MIN_FREQUENCY, MIN_SWEEP_DURATION, TWO_PI},
    FrequencySweep,
};
use crate::signal::Signal;

/// Exponential frequency sweep
///
/// Frequency varies exponentially with time
#[derive(Debug, Clone)]
pub struct ExponentialSweep {
    start_frequency: f64,
    stop_frequency: f64,
    duration: f64,
    amplitude: f64,
    exp_rate: f64,
}

impl ExponentialSweep {
    /// Create new exponential sweep - USING all parameters
    #[must_use]
    pub fn new(start_freq: f64, stop_freq: f64, duration: f64, amplitude: f64) -> Self {
        assert!(start_freq > MIN_FREQUENCY, "Start frequency too low");
        assert!(stop_freq > MIN_FREQUENCY, "Stop frequency too low");
        assert!(duration > MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        // Calculate exponential rate
        let exp_rate = (stop_freq / start_freq).ln() / duration;

        Self {
            start_frequency: start_freq,
            stop_frequency: stop_freq,
            duration,
            amplitude,
            exp_rate,
        }
    }
}

impl Signal for ExponentialSweep {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        let phase = FrequencySweep::phase(self, t);
        self.amplitude * phase.sin()
    }

    fn duration(&self) -> Option<f64> {
        Some(self.duration)
    }

    fn frequency(&self, t: f64) -> f64 {
        self.instantaneous_frequency(t)
    }

    fn phase(&self, t: f64) -> f64 {
        FrequencySweep::phase(self, t)
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

impl FrequencySweep for ExponentialSweep {
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        self.start_frequency * (self.exp_rate * t).exp()
    }

    fn phase(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        if self.exp_rate.abs() < EPSILON {
            TWO_PI * self.start_frequency * t
        } else {
            TWO_PI * self.start_frequency / self.exp_rate * ((self.exp_rate * t).exp() - 1.0)
        }
    }

    fn sweep_rate(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        self.instantaneous_frequency(t) * self.exp_rate
    }

    fn start_frequency(&self) -> f64 {
        self.start_frequency
    }

    fn stop_frequency(&self) -> f64 {
        self.stop_frequency
    }

    fn duration(&self) -> f64 {
        self.duration
    }
}
