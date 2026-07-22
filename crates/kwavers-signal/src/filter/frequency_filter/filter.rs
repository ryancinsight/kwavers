//! `FrequencyFilter` — FFT-based frequency-domain filter.

use crate::Filter;
use apollo::{fft_1d_array, ifft_1d_array, Complex64};
use kwavers_core::error::KwaversResult;

/// Frequency-domain filter using ideal (brick-wall) FFT-based frequency responses.
///
/// Provides bandpass, lowpass, and highpass filtering via:
/// 1. Forward FFT: `X`K` = FFT(x`N`)`
/// 2. Apply response: `Y`K` = X`K` · H`K`` where `H`K` ∈ {0, 1}`
/// 3. Inverse FFT: `y`N` = IFFT(Y`K`)`
///
/// ## Frequency Resolution
///
/// For a signal of length N sampled at interval dt:
/// - `df = 1 / (N · dt)` Hz/bin
/// - `f_nyq = 1 / (2 · dt)` Hz
///
/// ## Properties
///
/// - Time complexity: O(N log N)
/// - Zero-phase response (no phase distortion)
/// - Assumes periodic signal (boundary effects possible)
///
/// ## References
///
/// - Cooley & Tukey (1965). Math. Comput., 19(90), 297–301.
/// - Oppenheim & Schafer (2009). Discrete-Time Signal Processing, 3rd ed.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrequencyFilter;

impl FrequencyFilter {
    /// Create a new `FrequencyFilter`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Bandpass filter: pass `low_freq ≤ f ≤ high_freq`.
    ///
    /// # Arguments
    /// - `signal`: Input time-domain samples.
    /// - `dt`: Sampling interval (s).
    /// - `low_freq`: Lower cutoff (Hz).
    /// - `high_freq`: Upper cutoff (Hz).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn bandpass(
        &self,
        signal: &[f64],
        dt: f64,
        low_freq: f64,
        high_freq: f64,
    ) -> KwaversResult<Vec<f64>> {
        self.apply_filter(signal, dt, |f| f >= low_freq && f <= high_freq)
    }

    /// Lowpass filter: pass `f ≤ cutoff`.
    ///
    /// # Arguments
    /// - `signal`: Input time-domain samples.
    /// - `dt`: Sampling interval (s).
    /// - `cutoff`: Cutoff frequency (Hz).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn lowpass(&self, signal: &[f64], dt: f64, cutoff: f64) -> KwaversResult<Vec<f64>> {
        self.apply_filter(signal, dt, |f| f <= cutoff)
    }

    /// Highpass filter: pass `f ≥ cutoff`.
    ///
    /// # Arguments
    /// - `signal`: Input time-domain samples.
    /// - `dt`: Sampling interval (s).
    /// - `cutoff`: Cutoff frequency (Hz).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn highpass(&self, signal: &[f64], dt: f64, cutoff: f64) -> KwaversResult<Vec<f64>> {
        self.apply_filter(signal, dt, |f| f >= cutoff)
    }

    /// Time-domain windowing: zero out samples outside `[t_min, t_max]`.
    ///
    /// # Arguments
    /// - `signal`: Input samples (consumed).
    /// - `dt`: Sampling interval (s).
    /// - `time_window`: `(t_min, t_max)` — closed interval in seconds.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

    /// Apply an arbitrary frequency response function.
    ///
    /// `resp(f)` returns `true` for frequencies to pass, `false` to zero out.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_filter<F>(&self, signal: &[f64], dt: f64, resp: F) -> KwaversResult<Vec<f64>>
    where
        F: Fn(f64) -> bool,
    {
        let n = signal.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let input = leto::Array1::from_shape_vec([n], signal.to_vec())
            .expect("frequency-filter input length must match Leto shape");
        let mut spectrum = fft_1d_array(&input);

        let df = 1.0 / (n as f64 * dt);
        for (i, val) in spectrum.iter_mut().enumerate() {
            // Bins 0..=N/2 → positive frequencies; N/2+1..N → mirror (negative).
            let freq = if i <= n / 2 {
                i as f64 * df
            } else {
                (n - i) as f64 * df
            };
            if !resp(freq) {
                *val = Complex64::new(0.0, 0.0);
            }
        }

        Ok(ifft_1d_array(&spectrum).into_vec())
    }
}

impl Filter for FrequencyFilter {
    /// Pass-through implementation of the `Filter` trait.
    ///
    /// Use `bandpass`, `lowpass`, or `highpass` for actual filtering.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply(&self, signal: &[f64], _dt: f64) -> KwaversResult<Vec<f64>> {
        Ok(signal.to_vec())
    }
}
