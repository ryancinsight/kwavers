// signal/chirp.rs

use crate::signal::{
    amplitude::{Amplitude, ConstantAmplitude},
    frequency::Frequency,
    phase::{ConstantPhase, Phase},
    Signal,
};
use std::fmt::Debug;

#[derive(Debug, Clone))]
pub struct ChirpSignal {
    amplitude: Box<dyn Amplitude>,
    frequency: Box<dyn Frequency>,
    phase: Box<dyn Phase>,
    duration: f64,
    start_freq: f64,
    end_freq: f64,
}

impl ChirpSignal {
    pub fn new(start_freq: f64, end_freq: f64, duration: f64, amplitude: f64, phase: f64) -> Self {
        assert!(start_freq > 0.0 && end_freq > 0.0 && duration > 0.0 && amplitude >= 0.0);
        Self {
            amplitude: Box::new(ConstantAmplitude::new(amplitude)),
            frequency: Box::new(crate::signal::frequency::QuadraticSweep::new(
                start_freq, end_freq, duration, 2.0,
            )),
            phase: Box::new(ConstantPhase::new(phase)),
            duration,
            start_freq,
            end_freq,
        }
    }
}

impl Signal for ChirpSignal {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            0.0
        } else {
            let amp = self.amplitude.amplitude(t);
            let freq = self.frequency.frequency(t);
            let phase = self.phase.phase(t) + self.integrated_phase(t);
            amp * (2.0 * std::f64::consts::PI * freq * t + phase).sin()
        }
    }

    fn duration(&self) -> Option<f64> {
        Some(self.duration)
    }

    fn frequency(&self, t: f64) -> f64 {
        self.frequency.frequency(t)
    }

    fn phase(&self, t: f64) -> f64 {
        self.phase.phase(t) + self.integrated_phase(t)
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

impl ChirpSignal {
    fn integrated_phase(&self, t: f64) -> f64 {
        if t >= self.duration {
            self.start_freq * self.duration
                + (self.end_freq - self.start_freq) * self.duration / 3.0
        } else {
            self.start_freq * t
                + (self.end_freq - self.start_freq) * t.powi(3) / (3.0 * self.duration.powi(2))
        }
    }
}
