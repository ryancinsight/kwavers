//! Tone burst signal implementation

use crate::signal::Signal;
use std::f64::consts::PI;

use super::MAX_TONE_BURST_CYCLES;

/// Tone burst signal - windowed sinusoid with specific number of cycles
///
/// Common in ultrasound applications for controlled excitation
#[derive(Debug, Clone)]
pub struct ToneBurst {
    center_frequency: f64,
    num_cycles: f64,
    start_time: f64,
    amplitude: f64,
    phase: f64,
    window_type: WindowType,
}

#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    Gaussian,
    Tukey { alpha: f64 }, // alpha = 0: rectangular, alpha = 1: Hann
}

impl ToneBurst {
    pub fn new(center_frequency: f64, num_cycles: f64, start_time: f64, amplitude: f64) -> Self {
        assert!(center_frequency > 0.0, "Center frequency must be positive");
        assert!(num_cycles > 0.0 && num_cycles <= MAX_TONE_BURST_CYCLES as f64);
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            center_frequency,
            num_cycles,
            start_time,
            amplitude,
            phase: 0.0,
            window_type: WindowType::Hann,
        }
    }

    pub fn with_window(mut self, window_type: WindowType) -> Self {
        self.window_type = window_type;
        self
    }

    fn duration_seconds(&self) -> f64 {
        self.num_cycles / self.center_frequency
    }

    fn window(&self, t: f64) -> f64 {
        let relative_time = t - self.start_time;
        let duration = self.duration_seconds();

        if relative_time < 0.0 || relative_time > duration {
            return 0.0;
        }

        let normalized_time = relative_time / duration;

        match self.window_type {
            WindowType::Rectangular => 1.0,

            WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * normalized_time).cos()),

            WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * normalized_time).cos(),

            WindowType::Blackman => {
                0.42 - 0.5 * (2.0 * PI * normalized_time).cos()
                    + 0.08 * (4.0 * PI * normalized_time).cos()
            }

            WindowType::Gaussian => {
                let sigma = 0.4; // Standard deviation (adjustable)
                let arg = (normalized_time - 0.5) / sigma;
                (-0.5 * arg * arg).exp()
            }

            WindowType::Tukey { alpha } => {
                if normalized_time < alpha / 2.0 {
                    0.5 * (1.0 + (2.0 * PI * normalized_time / alpha - PI).cos())
                } else if normalized_time < 1.0 - alpha / 2.0 {
                    1.0
                } else {
                    0.5 * (1.0 + (2.0 * PI * (normalized_time - 1.0) / alpha + PI).cos())
                }
            }
        }
    }
}

impl Signal for ToneBurst {
    fn amplitude(&self, t: f64) -> f64 {
        let window = self.window(t);
        let carrier = (2.0 * PI * self.center_frequency * t + self.phase).sin();
        self.amplitude * window * carrier
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.center_frequency
    }

    fn phase(&self, _t: f64) -> f64 {
        self.phase
    }

    fn duration(&self) -> Option<f64> {
        Some(self.duration_seconds())
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}
