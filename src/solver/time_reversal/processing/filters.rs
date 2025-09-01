//! Signal Filtering and Processing Module
//!
//! Implements frequency filtering and signal processing for time-reversal reconstruction.

use crate::error::KwaversResult;
use log::debug;
use rustfft::{num_complex::Complex, FftPlanner};

/// Frequency filter for time-reversal signals
pub struct FrequencyFilter {
    fft_planner: FftPlanner<f64>,
}

impl std::fmt::Debug for FrequencyFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrequencyFilter")
            .field("fft_planner", &"<FftPlanner>")
            .finish()
    }
}

impl FrequencyFilter {
    /// Create a new frequency filter
    pub fn new() -> Self {
        Self {
            fft_planner: FftPlanner::new(),
        }
    }

    /// Apply frequency filtering to a signal
    pub fn apply_bandpass(
        &mut self,
        signal: Vec<f64>,
        dt: f64,
        frequency_range: (f64, f64),
    ) -> KwaversResult<Vec<f64>> {
        let n = signal.len();
        let (f_min, f_max) = frequency_range;

        // Convert to complex for FFT
        let mut complex_signal: Vec<Complex<f64>> =
            signal.into_iter().map(|x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        let fft = self.fft_planner.plan_fft_forward(n);
        fft.process(&mut complex_signal);

        // Apply frequency filter
        let df = 1.0 / (n as f64 * dt);
        for (i, val) in complex_signal.iter_mut().enumerate() {
            let freq = if i <= n / 2 {
                i as f64 * df
            } else {
                (n - i) as f64 * df
            };

            // Butterworth-like filter response
            if freq < f_min || freq > f_max {
                *val = Complex::new(0.0, 0.0);
            }
        }

        // Inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(n);
        ifft.process(&mut complex_signal);

        // Convert back to real and normalize
        let filtered_signal: Vec<f64> = complex_signal
            .into_iter()
            .map(|x| x.re / n as f64)
            .collect();

        debug!(
            "Applied frequency filter: [{:.1} Hz, {:.1} Hz]",
            f_min, f_max
        );
        Ok(filtered_signal)
    }

    /// Apply time windowing to a signal
    pub fn apply_time_window(
        &self,
        signal: Vec<f64>,
        dt: f64,
        time_window: (f64, f64),
    ) -> KwaversResult<Vec<f64>> {
        let (t_min, t_max) = time_window;
        let windowed: Vec<f64> = signal
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
            .collect();

        debug!("Applied time window: [{:.3}s, {:.3}s]", t_min, t_max);
        Ok(windowed)
    }
}

impl Default for FrequencyFilter {
    fn default() -> Self {
        Self::new()
    }
}
