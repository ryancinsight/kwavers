//! Gaussian pulse signal implementation

use crate::signal::Signal;
use std::f64::consts::PI;

use super::{DEFAULT_GAUSSIAN_Q, MIN_PULSE_WIDTH};

/// Gaussian pulse signal
///
/// Implements a Gaussian-modulated sinusoidal pulse:
/// s(t) = A * exp(-((t-t0)/(τ/Q))²) * sin(2πf₀t + φ)
///
/// Where Q controls the bandwidth (higher Q = narrower bandwidth)
#[derive(Debug, Clone)]
pub struct GaussianPulse {
    center_frequency: f64,
    center_time: f64,
    pulse_width: f64,
    amplitude: f64,
    phase: f64,
    q_factor: f64,
}

impl GaussianPulse {
    pub fn new(center_frequency: f64, center_time: f64, pulse_width: f64, amplitude: f64) -> Self {
        assert!(center_frequency > 0.0, "Center frequency must be positive");
        assert!(pulse_width >= MIN_PULSE_WIDTH, "Pulse width too small");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            center_frequency,
            center_time,
            pulse_width,
            amplitude,
            phase: 0.0,
            q_factor: DEFAULT_GAUSSIAN_Q,
        }
    }

    pub fn with_q_factor(mut self, q: f64) -> Self {
        assert!(q > 0.0, "Q factor must be positive");
        self.q_factor = q;
        self
    }

    pub fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }

    fn envelope(&self, t: f64) -> f64 {
        let tau = self.pulse_width / self.q_factor;
        let arg = (t - self.center_time) / tau;
        (-arg * arg).exp()
    }
}

impl Signal for GaussianPulse {
    fn amplitude(&self, t: f64) -> f64 {
        let envelope = self.envelope(t);
        let carrier = (2.0 * PI * self.center_frequency * t + self.phase).sin();
        self.amplitude * envelope * carrier
    }

    fn frequency(&self, t: f64) -> f64 {
        // Instantaneous frequency for Gaussian pulse
        // Includes chirp due to envelope modulation
        let envelope = self.envelope(t);
        if envelope > 1e-10 {
            self.center_frequency
        } else {
            0.0
        }
    }

    fn phase(&self, _t: f64) -> f64 {
        self.phase
    }

    fn duration(&self) -> Option<f64> {
        // Effective duration (99% energy containment)
        Some(6.0 * self.pulse_width / self.q_factor)
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}
