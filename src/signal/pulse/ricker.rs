//! Ricker wavelet implementation

use crate::signal::Signal;
use std::f64::consts::PI;

/// Ricker wavelet (Mexican hat wavelet)
///
/// Commonly used in seismic and ultrasound applications
/// Second derivative of Gaussian function
///
/// Reference: Ricker, N. (1953). "The form and laws of propagation of seismic wavelets"
#[derive(Debug, Clone)]
pub struct RickerWavelet {
    peak_frequency: f64,
    peak_time: f64,
    amplitude: f64,
}

impl RickerWavelet {
    pub fn new(peak_frequency: f64, peak_time: f64, amplitude: f64) -> Self {
        assert!(peak_frequency > 0.0, "Peak frequency must be positive");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            peak_frequency,
            peak_time,
            amplitude,
        }
    }

    /// Compute the Ricker wavelet value
    /// r(t) = A * (1 - 2π²f²τ²) * exp(-π²f²τ²)
    /// where τ = t - t_peak
    fn ricker_value(&self, t: f64) -> f64 {
        let tau = t - self.peak_time;
        let f = self.peak_frequency;
        let arg = PI * f * tau;
        let arg_squared = arg * arg;

        (1.0 - 2.0 * arg_squared) * (-arg_squared).exp()
    }
}

impl Signal for RickerWavelet {
    fn amplitude(&self, t: f64) -> f64 {
        self.amplitude * self.ricker_value(t)
    }

    fn frequency(&self, t: f64) -> f64 {
        // Instantaneous frequency of Ricker wavelet
        // Peak at center, decreases away from center
        let tau = (t - self.peak_time).abs();
        let decay_factor = (-PI * PI * self.peak_frequency * self.peak_frequency * tau * tau).exp();
        self.peak_frequency * decay_factor
    }

    fn phase(&self, _t: f64) -> f64 {
        0.0 // Ricker wavelet is real-valued
    }

    fn duration(&self) -> Option<f64> {
        // Effective duration (99% energy)
        Some(4.0 / self.peak_frequency)
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}
