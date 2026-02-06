// signal/sine_wave.rs

use crate::domain::signal::{
    amplitude::{Amplitude, ConstantAmplitude},
    frequency::{ConstantFrequency, Frequency},
    phase::{ConstantPhase, Phase},
    Signal,
};

#[derive(Debug, Clone)]
pub struct SineWave {
    amplitude: Box<dyn Amplitude>,
    frequency: Box<dyn Frequency>,
    phase: Box<dyn Phase>,
}

impl SineWave {
    pub fn new_with_components<
        A: Amplitude + 'static,
        F: Frequency + 'static,
        P: Phase + 'static,
    >(
        amplitude: A,
        frequency: F,
        phase: P,
    ) -> Self {
        Self {
            amplitude: Box::new(amplitude),
            frequency: Box::new(frequency),
            phase: Box::new(phase),
        }
    }

    #[must_use]
    pub fn new(frequency: f64, amplitude: f64, phase: f64) -> Self {
        Self::new_with_components(
            ConstantAmplitude::new(amplitude),
            ConstantFrequency::new(frequency),
            ConstantPhase::new(phase),
        )
    }

    #[must_use]
    pub fn new_180khz(amplitude: f64) -> Self {
        Self::new(180000.0, amplitude, 0.0)
    }

    fn frequency_hz_checked(&self, t: f64) -> f64 {
        let f0 = self.frequency.frequency(0.0);
        let ft = self.frequency.frequency(t);
        let tol = 1e-12 * f0.abs().max(1.0);
        assert!(
            (ft - f0).abs() <= tol,
            "Time-varying frequency is not supported for waveforms"
        );
        f0
    }
}

impl Signal for SineWave {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 {
            0.0
        } else {
            let amp = self.amplitude.amplitude(t);
            let freq = self.frequency_hz_checked(t);
            let phase = self.phase.phase(t);
            amp * (2.0 * std::f64::consts::PI * freq * t + phase).sin()
        }
    }

    fn frequency(&self, t: f64) -> f64 {
        self.frequency_hz_checked(t)
    }
    fn phase(&self, t: f64) -> f64 {
        self.phase.phase(t)
    }
    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}
