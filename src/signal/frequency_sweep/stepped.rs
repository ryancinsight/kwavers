// frequency_sweep/stepped.rs - Stepped frequency sweep

use super::{
    constants::{MIN_FREQUENCY, MIN_SWEEP_DURATION, TWO_PI},
    FrequencySweep,
};
use crate::signal::Signal;

/// Stepped frequency sweep
///
/// Frequency changes in discrete steps
#[derive(Debug, Clone)]
pub struct SteppedSweep {
    start_frequency: f64,
    stop_frequency: f64,
    duration: f64,
    num_steps: usize,
    amplitude: f64,
    step_duration: f64,
    frequency_step: f64,
}

impl SteppedSweep {
    /// Create new stepped sweep - USING all parameters
    #[must_use]
    pub fn new(
        start_freq: f64,
        stop_freq: f64,
        duration: f64,
        num_steps: usize,
        amplitude: f64,
    ) -> Self {
        assert!(start_freq > MIN_FREQUENCY, "Start frequency too low");
        assert!(stop_freq > MIN_FREQUENCY, "Stop frequency too low");
        assert!(duration > MIN_SWEEP_DURATION, "Duration too short");
        assert!(num_steps > 0, "Must have at least one step");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        let step_duration = duration / num_steps as f64;
        let frequency_step = (stop_freq - start_freq) / (num_steps - 1).max(1) as f64;

        Self {
            start_frequency: start_freq,
            stop_frequency: stop_freq,
            duration,
            num_steps,
            amplitude,
            step_duration,
            frequency_step,
        }
    }

    /// Get current step index
    fn get_step_index(&self, t: f64) -> usize {
        if t <= 0.0 {
            0
        } else if t >= self.duration {
            self.num_steps - 1
        } else {
            ((t / self.step_duration) as usize).min(self.num_steps - 1)
        }
    }
}

impl Signal for SteppedSweep {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        let freq = self.instantaneous_frequency(t);
        let phase = TWO_PI * freq * t;
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

impl FrequencySweep for SteppedSweep {
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        let step = self.get_step_index(t);
        self.start_frequency + self.frequency_step * step as f64
    }

    fn phase(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        let step = self.get_step_index(t);
        let mut phase = 0.0;

        // Sum phase from previous steps
        for i in 0..step {
            let freq = self.start_frequency + self.frequency_step * i as f64;
            phase += TWO_PI * freq * self.step_duration;
        }

        // Add phase from current step
        let current_freq = self.start_frequency + self.frequency_step * step as f64;
        let time_in_step = t - step as f64 * self.step_duration;
        phase += TWO_PI * current_freq * time_in_step;

        phase
    }

    fn sweep_rate(&self, _t: f64) -> f64 {
        // Stepped sweep has zero instantaneous sweep rate
        // (infinite at transitions, zero elsewhere)
        0.0
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
