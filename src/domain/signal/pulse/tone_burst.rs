//! Tone burst signal implementation

use crate::domain::core::error::{KwaversError, KwaversResult};
use crate::domain::signal::window::{window_value, WindowType};
use crate::domain::signal::Signal;
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

impl ToneBurst {
    #[must_use]
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

    pub fn try_new(
        center_frequency: f64,
        num_cycles: f64,
        start_time: f64,
        amplitude: f64,
    ) -> KwaversResult<Self> {
        if !center_frequency.is_finite() || center_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "center_frequency must be finite and > 0, got {center_frequency}"
            )));
        }
        if !num_cycles.is_finite() || num_cycles <= 0.0 || num_cycles > MAX_TONE_BURST_CYCLES as f64
        {
            return Err(KwaversError::InvalidInput(format!(
                "num_cycles must be finite, in (0, {}], got {num_cycles}",
                MAX_TONE_BURST_CYCLES
            )));
        }
        if !start_time.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "start_time must be finite, got {start_time}"
            )));
        }
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "amplitude must be finite and >= 0, got {amplitude}"
            )));
        }

        Ok(Self {
            center_frequency,
            num_cycles,
            start_time,
            amplitude,
            phase: 0.0,
            window_type: WindowType::Hann,
        })
    }

    #[must_use]
    pub fn with_window(mut self, window_type: WindowType) -> Self {
        self.window_type = window_type;
        self
    }

    #[must_use]
    pub fn with_phase(mut self, phase: f64) -> Self {
        assert!(phase.is_finite(), "Phase must be finite");
        self.phase = phase;
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
        window_value(self.window_type, normalized_time)
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
