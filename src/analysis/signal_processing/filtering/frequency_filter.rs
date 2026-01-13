//! Frequency-Domain Signal Filtering
//!
//! This module provides FFT-based frequency-domain filtering algorithms for
//! acoustic and ultrasound signals. It implements bandpass, lowpass, and highpass
//! filters using the Fast Fourier Transform (FFT).
//!
//! ## Architecture
//!
//! This module resides in the **analysis layer** (`analysis::signal_processing::filtering`)
//! because filtering is a signal processing operation, not a domain primitive.
//!
//! ### Layer Placement Rationale
//!
//! - **Domain Layer**: Contains the `Filter` trait (interface/contract)
//! - **Analysis Layer**: Contains filter implementations (algorithms)
//!
//! This separation follows the **Dependency Inversion Principle**:
//! - High-level modules depend on abstractions (Filter trait)
//! - Low-level implementations (FrequencyFilter) satisfy the abstraction
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
//! use kwavers::core::error::KwaversResult;
//!
//! fn example() -> KwaversResult<()> {
//!     let filter = FrequencyFilter::new();
//!
//!     // Sample signal at 10 kHz
//!     let dt = 1.0 / 10_000.0;
//!     let signal = vec![1.0, 0.5, -0.3, 0.8, -0.6]; // Example data
//!
//!     // Apply 1-3 kHz bandpass filter
//!     let filtered = filter.bandpass(&signal, dt, 1000.0, 3000.0)?;
//!
//!     // Apply 2 kHz lowpass filter
//!     let lowpassed = filter.lowpass(&signal, dt, 2000.0)?;
//!
//!     // Apply 500 Hz highpass filter
//!     let highpassed = filter.highpass(&signal, dt, 500.0)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Algorithm Details
//!
//! ### Frequency-Domain Filtering
//!
//! 1. **Forward FFT**: Transform signal to frequency domain
//!    ```text
//!    X[k] = FFT(x[n])
//!    ```
//!
//! 2. **Apply Frequency Response**: Zero out undesired frequencies
//!    ```text
//!    Y[k] = X[k] · H[k]
//!    where H[k] = 1 if frequency in passband, 0 otherwise
//!    ```
//!
//! 3. **Inverse FFT**: Transform back to time domain
//!    ```text
//!    y[n] = IFFT(Y[k])
//!    ```
//!
//! ### Frequency Calculation
//!
//! For a signal of length N sampled at interval dt:
//! - Frequency resolution: df = 1 / (N · dt)
//! - Nyquist frequency: f_nyq = 1 / (2 · dt)
//! - Frequency bins: f[k] = k · df for k = 0, 1, ..., N/2
//!
//! ## Performance Characteristics
//!
//! - **Time Complexity**: O(N log N) due to FFT
//! - **Space Complexity**: O(N) for FFT buffers
//! - **Numerical Stability**: Excellent (FFT is numerically stable)
//!
//! ## Limitations
//!
//! 1. **Brick-Wall Response**: Ideal frequency response with sharp cutoffs
//!    - Can introduce ringing artifacts (Gibbs phenomenon)
//!    - Consider windowing or IIR filters for smoother response
//!
//! 2. **Zero-Phase**: FFT filtering introduces no phase distortion
//!    - Good for preserving waveform shape
//!    - Cannot be used for real-time causal filtering
//!
//! 3. **Boundary Effects**: Assumes signal is periodic
//!    - Consider zero-padding or windowing at boundaries
//!
//! ## References
//!
//! - Cooley, J. W., & Tukey, J. W. (1965). "An algorithm for the machine calculation
//!   of complex Fourier series." *Mathematics of Computation*, 19(90), 297-301.
//! - Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing*
//!   (3rd ed.). Pearson.
//!
//! ## Migration Note
//!
//! This implementation was moved from `domain::signal::filter` to
//! `analysis::signal_processing::filtering` in Sprint 188 Phase 3 to enforce
//! proper architectural layering (domain = primitives, analysis = algorithms).

use crate::core::error::KwaversResult;
use crate::domain::signal::Filter;
use rustfft::{num_complex::Complex, FftPlanner};

/// Frequency-domain filter implementation using FFT
///
/// Provides bandpass, lowpass, and highpass filtering using the Fast Fourier Transform.
/// This implementation uses ideal (brick-wall) frequency responses, which provide
/// sharp cutoffs but may introduce ringing artifacts.
///
/// # Examples
///
/// ```rust,no_run
/// use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
///
/// let filter = FrequencyFilter::new();
/// let signal = vec![1.0, 0.5, -0.3, 0.8, -0.6];
/// let dt = 0.0001; // 100 μs sampling interval (10 kHz sample rate)
///
/// // Apply 1-5 kHz bandpass filter
/// let filtered = filter.bandpass(&signal, dt, 1000.0, 5000.0).unwrap();
/// ```
///
/// # Performance
///
/// - Time complexity: O(N log N) where N is signal length
/// - Space complexity: O(N) for FFT buffers
/// - Zero-phase response (no phase distortion)
///
/// # Thread Safety
///
/// `FrequencyFilter` is stateless and can be safely shared across threads.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrequencyFilter;

impl FrequencyFilter {
    /// Create a new frequency-domain filter
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
    ///
    /// let filter = FrequencyFilter::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Apply a bandpass filter to a signal
    ///
    /// Passes frequencies between `low_freq` and `high_freq`, attenuates all others.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input time-domain signal
    /// * `dt` - Time step (sampling interval in seconds)
    /// * `low_freq` - Lower cutoff frequency (Hz)
    /// * `high_freq` - Upper cutoff frequency (Hz)
    ///
    /// # Returns
    ///
    /// Filtered signal with same length as input
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
    ///
    /// let filter = FrequencyFilter::new();
    /// let signal = vec![1.0; 1000];
    /// let dt = 1.0 / 10_000.0; // 10 kHz sample rate
    ///
    /// // Pass 1-3 kHz frequencies
    /// let filtered = filter.bandpass(&signal, dt, 1000.0, 3000.0).unwrap();
    /// ```
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
    ///
    /// Passes frequencies below `cutoff`, attenuates higher frequencies.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input time-domain signal
    /// * `dt` - Time step (sampling interval in seconds)
    /// * `cutoff` - Cutoff frequency (Hz)
    ///
    /// # Returns
    ///
    /// Filtered signal with same length as input
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
    ///
    /// let filter = FrequencyFilter::new();
    /// let signal = vec![1.0; 1000];
    /// let dt = 1.0 / 10_000.0; // 10 kHz sample rate
    ///
    /// // Remove frequencies above 2 kHz
    /// let filtered = filter.lowpass(&signal, dt, 2000.0).unwrap();
    /// ```
    pub fn lowpass(&self, signal: &[f64], dt: f64, cutoff: f64) -> KwaversResult<Vec<f64>> {
        self.apply_filter(signal, dt, |f| f <= cutoff)
    }

    /// Apply a highpass filter to a signal
    ///
    /// Passes frequencies above `cutoff`, attenuates lower frequencies.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input time-domain signal
    /// * `dt` - Time step (sampling interval in seconds)
    /// * `cutoff` - Cutoff frequency (Hz)
    ///
    /// # Returns
    ///
    /// Filtered signal with same length as input
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
    ///
    /// let filter = FrequencyFilter::new();
    /// let signal = vec![1.0; 1000];
    /// let dt = 1.0 / 10_000.0; // 10 kHz sample rate
    ///
    /// // Remove DC and low frequencies below 500 Hz
    /// let filtered = filter.highpass(&signal, dt, 500.0).unwrap();
    /// ```
    pub fn highpass(&self, signal: &[f64], dt: f64, cutoff: f64) -> KwaversResult<Vec<f64>> {
        self.apply_filter(signal, dt, |f| f >= cutoff)
    }

    /// Apply time-domain windowing to a signal
    ///
    /// Zeros out signal samples outside the specified time window.
    /// Useful for isolating specific time-of-flight segments.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input time-domain signal (consumed)
    /// * `dt` - Time step (sampling interval in seconds)
    /// * `time_window` - (t_min, t_max) time window in seconds
    ///
    /// # Returns
    ///
    /// Windowed signal with samples outside window set to zero
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
    ///
    /// let filter = FrequencyFilter::new();
    /// let signal = vec![1.0; 1000];
    /// let dt = 1.0e-6; // 1 μs sample rate
    ///
    /// // Keep only 10-20 μs time window
    /// let windowed = filter.apply_time_window(signal, dt, (10.0e-6, 20.0e-6)).unwrap();
    /// ```
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

    /// Generic internal filtering logic
    ///
    /// Applies an arbitrary frequency response function to the signal.
    ///
    /// # Algorithm
    ///
    /// 1. Transform signal to frequency domain via FFT
    /// 2. Apply frequency response: Y[k] = X[k] if resp(f[k]), else 0
    /// 3. Transform back to time domain via IFFT
    ///
    /// # Arguments
    ///
    /// * `signal` - Input time-domain signal
    /// * `dt` - Time step (sampling interval in seconds)
    /// * `resp` - Frequency response function: f(Hz) -> bool (pass/reject)
    fn apply_filter<F>(&self, signal: &[f64], dt: f64, resp: F) -> KwaversResult<Vec<f64>>
    where
        F: Fn(f64) -> bool,
    {
        let n = signal.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Create FFT planners
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert real signal to complex
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT: time domain -> frequency domain
        fft.process(&mut complex_signal);

        // Apply frequency response
        let df = 1.0 / (n as f64 * dt); // Frequency resolution (Hz)
        for (i, val) in complex_signal.iter_mut().enumerate() {
            // Calculate frequency for this bin
            // For FFT output: bins 0..N/2 are positive frequencies,
            // bins N/2+1..N are negative frequencies (or equivalently, high positive freqs)
            let freq = if i <= n / 2 {
                i as f64 * df // Positive frequencies: 0, df, 2df, ..., f_nyquist
            } else {
                (n - i) as f64 * df // Mirror for negative frequencies
            };

            // Zero out frequencies outside passband
            if !resp(freq) {
                *val = Complex::new(0.0, 0.0);
            }
        }

        // Inverse FFT: frequency domain -> time domain
        ifft.process(&mut complex_signal);

        // Extract real part and normalize (rustfft requires manual normalization)
        let norm_factor = 1.0 / n as f64;
        Ok(complex_signal.iter().map(|c| c.re * norm_factor).collect())
    }
}

impl Filter for FrequencyFilter {
    /// Apply default filter behavior (pass-through)
    ///
    /// This is a minimal implementation of the `Filter` trait.
    /// For actual filtering, use `bandpass()`, `lowpass()`, or `highpass()`.
    fn apply(&self, signal: &[f64], _dt: f64) -> KwaversResult<Vec<f64>> {
        // Default behavior: pass-through (no filtering)
        // In practice, users should call specific filter methods
        Ok(signal.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: Generate sine wave at given frequency
    fn sine_wave(freq: f64, sample_rate: f64, n_samples: usize) -> Vec<f64> {
        let dt = 1.0 / sample_rate;
        (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f64 * dt).sin())
            .collect()
    }

    /// Helper: Compute RMS (root-mean-square) of signal
    fn rms(signal: &[f64]) -> f64 {
        let sum_sq: f64 = signal.iter().map(|&x| x * x).sum();
        (sum_sq / signal.len() as f64).sqrt()
    }

    #[test]
    fn test_bandpass_passes_in_band_frequency() {
        let filter = FrequencyFilter::new();
        let sample_rate = 10_000.0; // 10 kHz
        let dt = 1.0 / sample_rate;
        let n = 1024;

        // Generate 1 kHz sine wave (in passband)
        let signal = sine_wave(1000.0, sample_rate, n);
        let original_rms = rms(&signal);

        // Apply 500-2000 Hz bandpass
        let filtered = filter.bandpass(&signal, dt, 500.0, 2000.0).unwrap();
        let filtered_rms = rms(&filtered);

        // Should preserve most of signal (some attenuation at FFT boundaries is OK)
        assert!(filtered_rms > 0.9 * original_rms);
    }

    #[test]
    fn test_bandpass_rejects_out_of_band_frequency() {
        let filter = FrequencyFilter::new();
        let sample_rate = 10_000.0;
        let dt = 1.0 / sample_rate;
        let n = 1024;

        // Generate 100 Hz sine wave (below passband)
        let signal = sine_wave(100.0, sample_rate, n);
        let original_rms = rms(&signal);

        // Apply 500-2000 Hz bandpass
        let filtered = filter.bandpass(&signal, dt, 500.0, 2000.0).unwrap();
        let filtered_rms = rms(&filtered);

        // Should strongly attenuate signal
        assert!(filtered_rms < 0.1 * original_rms);
    }

    #[test]
    fn test_lowpass_filters_high_frequencies() {
        let filter = FrequencyFilter::new();
        let sample_rate = 10_000.0;
        let dt = 1.0 / sample_rate;
        let n = 1024;

        // Generate 3 kHz sine wave (above cutoff)
        let signal = sine_wave(3000.0, sample_rate, n);

        // Apply 2 kHz lowpass
        let filtered = filter.lowpass(&signal, dt, 2000.0).unwrap();
        let filtered_rms = rms(&filtered);

        // Should strongly attenuate
        assert!(filtered_rms < 0.1);
    }

    #[test]
    fn test_highpass_filters_low_frequencies() {
        let filter = FrequencyFilter::new();
        let sample_rate = 10_000.0;
        let dt = 1.0 / sample_rate;
        let n = 1024;

        // Generate 100 Hz sine wave (below cutoff)
        let signal = sine_wave(100.0, sample_rate, n);

        // Apply 500 Hz highpass
        let filtered = filter.highpass(&signal, dt, 500.0).unwrap();
        let filtered_rms = rms(&filtered);

        // Should strongly attenuate
        assert!(filtered_rms < 0.1);
    }

    #[test]
    fn test_time_window_zeros_outside_window() {
        let filter = FrequencyFilter::new();
        let dt = 0.0001; // 100 μs
        let signal = vec![1.0; 100]; // 100 samples = 10 ms

        // Keep only 0.001-0.003 s (samples 10-30 inclusive)
        // Mathematical specification: closed interval [t_min, t_max]
        // Sample 10: t = 10 * 0.0001 = 0.001 s (included)
        // Sample 30: t = 30 * 0.0001 = 0.003 s (included, t == t_max)
        // Sample 31: t = 31 * 0.0001 = 0.0031 s (excluded, t > t_max)
        let windowed = filter
            .apply_time_window(signal, dt, (0.001, 0.003))
            .unwrap();

        // Check samples outside window are zero
        assert!(
            windowed[0..10].iter().all(|&x| x == 0.0),
            "Samples before window should be zero"
        );
        // Samples 10-30 (inclusive) should be 1.0 - closed interval [t_min, t_max]
        assert!(
            windowed[10..=30].iter().all(|&x| x == 1.0),
            "Samples within window [0.001, 0.003] should be 1.0"
        );
        assert!(
            windowed[31..].iter().all(|&x| x == 0.0),
            "Samples after window should be zero"
        );
    }

    #[test]
    fn test_filter_trait_implementation() {
        let filter = FrequencyFilter::new();
        let signal = vec![1.0, 2.0, 3.0];

        // Default Filter trait behavior should pass through
        let result = filter.apply(&signal, 0.001).unwrap();
        assert_eq!(result, signal);
    }

    #[test]
    fn test_empty_signal_handling() {
        let filter = FrequencyFilter::new();
        let empty: Vec<f64> = vec![];
        let dt = 0.001;

        let result = filter.bandpass(&empty, dt, 100.0, 1000.0).unwrap();
        assert_eq!(result.len(), 0);
    }
}
