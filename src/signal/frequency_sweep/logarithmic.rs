// frequency_sweep/logarithmic.rs - Logarithmic frequency sweep

use super::{constants::*, FrequencySweep};
use crate::signal::Signal;

/// Logarithmic frequency sweep
///
/// Frequency varies logarithmically with time:
/// f(t) = f₀ * (f₁/f₀)^(t/T)
#[derive(Debug, Clone)]
pub struct LogarithmicSweep {
    start_frequency: f64,
    stop_frequency: f64,
    duration: f64,
    amplitude: f64,
    log_ratio: f64,
}

impl LogarithmicSweep {
    /// Create new logarithmic sweep - USING all parameters
    pub fn new(start_freq: f64, stop_freq: f64, duration: f64, amplitude: f64) -> Self {
        assert!(start_freq > MIN_FREQUENCY, "Start frequency too low");
        assert!(stop_freq > MIN_FREQUENCY, "Stop frequency too low");
        assert!(duration > MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        let ratio = stop_freq / start_freq;
        assert!(
            ratio.abs() < MAX_FREQUENCY_RATIO,
            "Frequency ratio too large"
        );

        let log_ratio = ratio.ln();

        Self {
            start_frequency: start_freq,
            stop_frequency: stop_freq,
            duration,
            amplitude,
            log_ratio,
        }
    }
}

impl Signal for LogarithmicSweep {
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

impl FrequencySweep for LogarithmicSweep {
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        let normalized_time = t / self.duration;
        self.start_frequency * (self.log_ratio * normalized_time).exp()
    }

    fn phase(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        // Integral of frequency over time
        let normalized_time = t / self.duration;
        if self.log_ratio.abs() < EPSILON {
            // Linear approximation for small log ratio
            TWO_PI * self.start_frequency * t
        } else {
            TWO_PI * self.start_frequency * self.duration / self.log_ratio
                * ((self.log_ratio * normalized_time).exp() - 1.0)
        }
    }

    fn sweep_rate(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        let freq = self.instantaneous_frequency(t);
        freq * self.log_ratio / self.duration
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
