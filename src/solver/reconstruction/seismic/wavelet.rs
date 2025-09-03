//! Source wavelet implementations for seismic imaging

use super::constants::RICKER_TIME_SHIFT;
use std::f64::consts::PI;

/// Ricker wavelet generator for seismic sources
#[derive(Debug)]
pub struct RickerWavelet {
    /// Dominant frequency in Hz
    pub frequency: f64,
    /// Time shift for causality
    pub time_shift: f64,
}

impl RickerWavelet {
    /// Create a new Ricker wavelet with specified frequency
    #[must_use]
    pub fn new(frequency: f64) -> Self {
        Self {
            frequency,
            time_shift: RICKER_TIME_SHIFT / frequency,
        }
    }

    /// Evaluate the Ricker wavelet at time t
    ///
    /// The Ricker wavelet is defined as:
    /// w(t) = (1 - 2π²f²t'²) * exp(-π²f²t'²)
    /// where t' = t - `t_shift`
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        let t_shifted = t - self.time_shift;
        let arg = PI * self.frequency * t_shifted;
        let arg_squared = arg * arg;
        (1.0 - 2.0 * arg_squared) * (-arg_squared).exp()
    }

    /// Generate a time series of the wavelet
    #[must_use]
    pub fn generate_time_series(&self, dt: f64, n_samples: usize) -> Vec<f64> {
        (0..n_samples)
            .map(|i| self.evaluate(i as f64 * dt))
            .collect()
    }

    /// Compute the frequency spectrum of the Ricker wavelet
    ///
    /// The analytical spectrum is:
    /// W(f) = (2f²/√π*f₀³) * exp(-f²/f₀²)
    #[must_use]
    pub fn frequency_spectrum(&self, freq: f64) -> f64 {
        let f_ratio = freq / self.frequency;
        let f_ratio_squared = f_ratio * f_ratio;
        (2.0 * f_ratio_squared / PI.sqrt()) * (-f_ratio_squared).exp()
    }

    /// Get the peak frequency (which differs from dominant frequency)
    #[must_use]
    pub fn peak_frequency(&self) -> f64 {
        self.frequency / PI.sqrt()
    }
}

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
        arg.exp() / (self.sigma * (2.0 * PI).sqrt())
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
