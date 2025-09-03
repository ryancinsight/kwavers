// frequency_sweep/hyperbolic.rs - Hyperbolic frequency sweep

use super::{
    constants::{EPSILON, MIN_FREQUENCY, MIN_SWEEP_DURATION, SINGULARITY_AVOIDANCE_FACTOR, TWO_PI},
    FrequencySweep,
};
use crate::signal::Signal;

/// Hyperbolic frequency sweep
///
/// Frequency varies hyperbolically with time
#[derive(Debug, Clone)]
pub struct HyperbolicSweep {
    start_frequency: f64,
    stop_frequency: f64,
    duration: f64,
    amplitude: f64,
    hyperbolic_rate: f64,
}

impl HyperbolicSweep {
    /// Create new hyperbolic sweep - USING all parameters
    #[must_use]
    pub fn new(start_freq: f64, stop_freq: f64, duration: f64, amplitude: f64) -> Self {
        assert!(start_freq > MIN_FREQUENCY, "Start frequency too low");
        assert!(stop_freq > MIN_FREQUENCY, "Stop frequency too low");
        assert!(duration > MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        // Calculate hyperbolic rate avoiding singularity
        let t_safe = duration * SINGULARITY_AVOIDANCE_FACTOR;
        let hyperbolic_rate = (stop_freq - start_freq) * t_safe / duration;

        Self {
            start_frequency: start_freq,
            stop_frequency: stop_freq,
            duration,
            amplitude,
            hyperbolic_rate,
        }
    }
}

impl Signal for HyperbolicSweep {
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

impl FrequencySweep for HyperbolicSweep {
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        // Avoid singularity at t = duration
        let t_safe = t.min(self.duration * SINGULARITY_AVOIDANCE_FACTOR);
        let denominator = 1.0 - t_safe / self.duration;

        if denominator.abs() < EPSILON {
            self.stop_frequency
        } else {
            self.start_frequency / denominator
        }
    }

    fn phase(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        // Integral of 1/(1-t/T) is -T*ln(1-t/T)
        let t_safe = t.min(self.duration * SINGULARITY_AVOIDANCE_FACTOR);
        let normalized = t_safe / self.duration;

        if (1.0 - normalized).abs() < EPSILON {
            TWO_PI * self.stop_frequency * t_safe
        } else {
            -TWO_PI * self.start_frequency * self.duration * (1.0 - normalized).ln()
        }
    }

    fn sweep_rate(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        let freq = self.instantaneous_frequency(t);
        freq * freq / (self.start_frequency * self.duration)
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
