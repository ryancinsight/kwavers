//! Source wavelet implementations for seismic imaging.

use kwavers_core::constants::numerical::TWO_PI;
use std::f64::consts::PI;

/// Gaussian wavelet generator
#[derive(Debug)]
pub struct GaussianWavelet {
    /// Standard deviation in time domain
    pub sigma: f64,
    /// Time shift for causality
    pub time_shift: f64,
}

impl GaussianWavelet {
    /// Create a new Gaussian wavelet
    #[must_use]
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma,
            time_shift: 3.0 * sigma, // 3 sigma for causality
        }
    }

    /// Evaluate the Gaussian wavelet at time t
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        let t_shifted = t - self.time_shift;
        let arg = -0.5 * (t_shifted / self.sigma).powi(2);
        arg.exp() / (self.sigma * (TWO_PI).sqrt())
    }

    /// Generate a time series of the wavelet
    #[must_use]
    pub fn generate_time_series(&self, dt: f64, n_samples: usize) -> Vec<f64> {
        (0..n_samples)
            .map(|i| self.evaluate(i as f64 * dt))
            .collect()
    }
}

/// Ormsby wavelet (trapezoidal bandpass filter)
#[derive(Debug)]
pub struct OrmsbyWavelet {
    /// Low cut frequency
    pub f1: f64,
    /// Low pass frequency
    pub f2: f64,
    /// High pass frequency
    pub f3: f64,
    /// High cut frequency
    pub f4: f64,
}

impl OrmsbyWavelet {
    /// Create a new Ormsby wavelet with specified frequency band
    /// # Panics
    /// - Panics if assertion fails: `Frequencies must be in ascending order`.
    ///
    #[must_use]
    pub fn new(f1: f64, f2: f64, f3: f64, f4: f64) -> Self {
        assert!(
            f1 < f2 && f2 < f3 && f3 < f4,
            "Frequencies must be in ascending order"
        );
        Self { f1, f2, f3, f4 }
    }

    /// Evaluate the Ormsby wavelet at time t
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        if t.abs() < 1e-10 {
            // Handle singularity at t=0
            return 0.0;
        }

        let pi_t = PI * t;
        let sinc = |f: f64| (f * pi_t).sin() / pi_t;

        let a4 = self.f4.powi(2) * sinc(self.f4) / (self.f4 - self.f3);
        let a3 = self.f3.powi(2) * sinc(self.f3) / (self.f4 - self.f3);
        let a2 = self.f2.powi(2) * sinc(self.f2) / (self.f2 - self.f1);
        let a1 = self.f1.powi(2) * sinc(self.f1) / (self.f2 - self.f1);

        PI * (a4 - a3 - a2 + a1)
    }

    /// Generate a time series of the wavelet
    #[must_use]
    pub fn generate_time_series(&self, dt: f64, n_samples: usize) -> Vec<f64> {
        (0..n_samples)
            .map(|i| {
                let t = (i as f64 - n_samples as f64 / 2.0) * dt;
                self.evaluate(t)
            })
            .collect()
    }
}
