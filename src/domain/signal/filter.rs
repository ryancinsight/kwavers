//! Signal filtering and frequency domain processing
//!
//! This module provides tools for temporal filtering of signals,
//! including bandpass, lowpass, and highpass filters using FFT.

use crate::core::error::KwaversResult;
use rustfft::{num_complex::Complex, FftPlanner};
use std::fmt::Debug;

/// Trait for signal filtering operations
pub trait Filter: Debug + Send + Sync {
    /// Apply the filter to a time-domain signal
    fn apply(&self, signal: &[f64], dt: f64) -> KwaversResult<Vec<f64>>;
}

/// Frequency-domain filter implementation
#[derive(Debug, Default)]
pub struct FrequencyFilter {
    // We could pre-allocate planners here if performance becomes an issue,
    // but for now we'll prioritize a clean stateless-looking API.
}

impl FrequencyFilter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a bandpass filter to a signal
    pub fn bandpass(
        &self,
        signal: &[f64],
        dt: f64,
        low_freq: f64,
        high_freq: f64,
    ) -> KwaversResult<Vec<f64>> {
        self.apply_filter(signal, dt, |f| f >= low_freq && f <= high_freq)
    }

    /// Apply a lowpass filter to a signal
    pub fn lowpass(&self, signal: &[f64], dt: f64, cutoff: f64) -> KwaversResult<Vec<f64>> {
        self.apply_filter(signal, dt, |f| f <= cutoff)
    }

    /// Apply a highpass filter to a signal
    pub fn highpass(&self, signal: &[f64], dt: f64, cutoff: f64) -> KwaversResult<Vec<f64>> {
        self.apply_filter(signal, dt, |f| f >= cutoff)
    }

    /// Generic internal filtering logic
    fn apply_filter<F>(&self, signal: &[f64], dt: f64, resp: F) -> KwaversResult<Vec<f64>>
    where
        F: Fn(f64) -> bool,
    {
        let n = signal.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        fft.process(&mut complex_signal);

        // Apply response in frequency domain
        let df = 1.0 / (n as f64 * dt);
        for (i, val) in complex_signal.iter_mut().enumerate() {
            let freq = if i <= n / 2 {
                i as f64 * df
            } else {
                (n - i) as f64 * df
            };

            if !resp(freq) {
                *val = Complex::new(0.0, 0.0);
            }
        }

        // Inverse FFT
        ifft.process(&mut complex_signal);

        // Extract real part and normalize
        let norm_factor = 1.0 / n as f64;
        Ok(complex_signal.iter().map(|c| c.re * norm_factor).collect())
    }

    /// Apply simple time-domain windowing
    pub fn apply_time_window(
        &self,
        signal: Vec<f64>,
        dt: f64,
        time_window: (f64, f64),
    ) -> KwaversResult<Vec<f64>> {
        let (t_min, t_max) = time_window;
        Ok(signal
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                let t = i as f64 * dt;
                if t >= t_min && t <= t_max {
                    val
                } else {
                    0.0
                }
            })
            .collect())
    }
}

impl Filter for FrequencyFilter {
    fn apply(&self, signal: &[f64], _dt: f64) -> KwaversResult<Vec<f64>> {
        // Default to a wide pass-through if not specified, but this trait
        // usually expects a specific configuration.
        Ok(signal.to_vec())
    }
}
