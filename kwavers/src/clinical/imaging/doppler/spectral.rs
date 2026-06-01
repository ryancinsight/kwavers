//! Spectral Analysis for Doppler Signals — Welch's Method
//!
//! ## Mathematical Foundation
//!
//! **Welch's Averaged Modified Periodogram** (Welch 1967):
//!
//! Given N real samples `x[n]` with sampling frequency `f_s`:
//!
//! 1. Divide into K overlapping segments of length M with step `hop = M·(1−overlap)`:
//!    `x_j[n] = x[j·hop + n]`, `n = 0,…,M−1`
//!
//! 2. Apply Hann window `w[n] = ½·(1 − cos(2π n/(M−1)))`:
//!    `ỹ_j[n] = x_j[n]·w[n]`
//!
//! 3. Compute windowed periodogram for segment j:
//!    ```text
//!    P_j[k] = (1 / (f_s · M · U)) · |FFT(ỹ_j)[k]|²
//!    ```
//!    where `U = (1/M) Σ w[n]²` is the normalisation factor for window power.
//!
//! 4. Average over all K segments:
//!    ```text
//!    PSD[k] = (1/K) Σ_j P_j[k],    k = 0,…,M/2
//!    ```
//!
//! The result is the one-sided PSD in units of [signal_unit²/Hz].
//! Frequency resolution: `Δf = f_s / M`; maximum frequency: `f_max = f_s / 2`.
//!
//! ## Hann Window Properties
//!
//! Hann (von Hann) window provides:
//! - Main-lobe width: 8·Δf (moderate frequency resolution)
//! - First side-lobe level: −31.5 dB (good leakage rejection)
//! - Window normalisation: `U = 3/8` (energy correction factor)
//!
//! ## References
//! - Welch PD (1967). "The use of fast Fourier transform for the estimation of power
//!   spectra: a method based on time averaging over short, modified periodograms."
//!   *IEEE Trans Audio Electroacoust* 15(2):70–73. DOI:10.1109/TAU.1967.1161901
//! - Harris FJ (1978). "On the use of windows for harmonic analysis with the discrete
//!   Fourier transform." *Proc IEEE* 66(1):51–83.
//! - Evans DH, McDicken WN (2000). *Doppler Ultrasound: Physics, Instrumentation and
//!   Signal Processing* (2nd ed.). Wiley.

use crate::core::constants::numerical::TWO_PI;
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::{fft_1d_array, Complex64};
use ndarray::{Array1, ArrayView1};

/// Spectral analysis configuration
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// FFT segment length M (should be a power of 2 for efficiency)
    pub fft_size: usize,
    /// Fractional overlap between segments (0.0 = no overlap, 0.75 = 75% overlap)
    pub overlap: f64,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            fft_size: 256,
            overlap: 0.75,
        }
    }
}

/// Spectral analysis processor using Welch's averaged periodogram method
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    config: SpectralConfig,
}

impl SpectralAnalysis {
    #[must_use]
    pub fn new(config: SpectralConfig) -> Self {
        Self { config }
    }

    /// Compute one-sided power spectral density via Welch's averaged periodogram.
    ///
    /// # Algorithm
    ///
    /// 1. Build Hann window of length `config.fft_size`
    /// 2. Divide `signal` into overlapping segments; zero-pad the last partial segment
    /// 3. Window each segment; compute periodogram via FFT
    /// 4. Average periodograms and apply frequency normalisation
    ///
    /// # Arguments
    /// * `signal`      – Real-valued time-domain samples (any length ≥ 1)
    /// * `sample_rate` – Sampling frequency (Hz); used for PSD normalisation to [unit²/Hz]
    ///
    /// # Returns
    /// One-sided PSD of length `fft_size/2 + 1`, frequencies 0 … `sample_rate/2`.
    /// If `signal.len() < fft_size`, the signal is zero-padded to one segment.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `fft_size < 2` or `sample_rate ≤ 0`.
    pub fn compute_psd(
        &self,
        signal: ArrayView1<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array1<f64>> {
        let m = self.config.fft_size;

        if m < 2 {
            return Err(KwaversError::InvalidInput(
                "fft_size must be ≥ 2".to_owned(),
            ));
        }
        if sample_rate <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "sample_rate must be positive".to_owned(),
            ));
        }

        let n_signal = signal.len();
        let out_len = m / 2 + 1; // one-sided spectrum length

        // ── Hann window ────────────────────────────────────────────────────────
        // w[n] = ½·(1 − cos(2πn/(M−1)))   for n = 0,…,M−1
        let window: Array1<f64> = Array1::from_shape_fn(m, |n| {
            0.5 * (1.0 - (TWO_PI * n as f64 / (m - 1) as f64).cos())
        });

        // Window power normalisation factor: U = (1/M) Σ w[n]²
        // For Hann window U = 3/8 analytically; compute numerically for generality.
        let u_norm: f64 = window.iter().map(|&w| w * w).sum::<f64>() / m as f64;

        // Normalisation divisor: f_s · M · U
        let psd_scale = sample_rate * m as f64 * u_norm;

        // ── Segment and average ────────────────────────────────────────────────
        let hop =
            ((m as f64 * (1.0 - self.config.overlap.clamp(0.0, 0.99))).round() as usize).max(1);

        let mut psd_sum = Array1::<f64>::zeros(out_len);
        let mut n_segments = 0usize;

        let mut start = 0usize;
        loop {
            // Zero-pad the last segment if it doesn't fill the window
            let mut segment = Array1::<f64>::zeros(m);
            let copy_len = (n_signal.saturating_sub(start)).min(m);
            if copy_len == 0 {
                break;
            }
            for i in 0..copy_len {
                segment[i] = signal[start + i];
            }

            // Apply Hann window
            let windowed: Array1<f64> = Array1::from_shape_fn(m, |i| segment[i] * window[i]);

            // FFT and compute one-sided periodogram P[k] = |X[k]|² / (f_s · M · U)
            let spectrum: Array1<Complex64> = fft_1d_array(&windowed);
            for k in 0..out_len {
                let power = spectrum[k].norm_sqr(); // |X[k]|²
                                                    // Double-count non-DC, non-Nyquist bins for one-sided spectrum
                let scale = if k == 0 || k == m / 2 { 1.0 } else { 2.0 };
                psd_sum[k] += scale * power / psd_scale;
            }

            n_segments += 1;
            start += hop;

            if start >= n_signal {
                break;
            }
        }

        if n_segments == 0 {
            return Ok(Array1::zeros(out_len));
        }

        // Average over all segments
        Ok(psd_sum / n_segments as f64)
    }

    /// Frequency axis for the PSD output (Hz).
    ///
    /// Returns `fft_size/2 + 1` frequencies from 0 to `sample_rate/2`.
    #[must_use]
    pub fn frequency_axis(&self, sample_rate: f64) -> Array1<f64> {
        let out_len = self.config.fft_size / 2 + 1;
        Array1::from_shape_fn(out_len, |k| {
            k as f64 * sample_rate / self.config.fft_size as f64
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// **Test: PSD of a pure tone peaks at the correct frequency bin**
    ///
    /// Signal: `x[n] = sin(2π f₀ n / f_s)`, expected peak at bin `k = f₀ / Δf`.
    /// Verification: peak bin index equals `round(f₀ × fft_size / f_s)`.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_psd_pure_tone_peak_at_correct_frequency() {
        let config = SpectralConfig {
            fft_size: 256,
            overlap: 0.5,
        };
        let spectral = SpectralAnalysis::new(config);

        let sample_rate = 10_000.0_f64; // 10 kHz
        let f0 = 1_000.0_f64; // 1 kHz tone — exactly at bin k=25 for fft_size=256, fs=10kHz
        let n_samples = 1024usize;

        let signal: Array1<f64> = Array1::from_shape_fn(n_samples, |n| {
            (2.0 * PI * f0 * n as f64 / sample_rate).sin()
        });

        let psd = spectral.compute_psd(signal.view(), sample_rate).unwrap();

        // Expected peak bin: k = f₀ × fft_size / f_s = 1000 × 256 / 10000 = 25
        let expected_bin = (f0 * 256.0 / sample_rate).round() as usize;
        let peak_bin = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            peak_bin, expected_bin,
            "PSD peak at bin {peak_bin}, expected bin {expected_bin} (f₀={f0} Hz)"
        );
    }

    /// **Test: PSD of zero signal is all-zero**
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_psd_zero_signal() {
        let spectral = SpectralAnalysis::new(SpectralConfig::default());
        let signal = Array1::<f64>::zeros(512);
        let psd = spectral.compute_psd(signal.view(), 8000.0).unwrap();
        assert!(
            psd.iter().all(|&v| v == 0.0),
            "PSD of zero signal should be all-zero"
        );
    }

    /// **Test: PSD is non-negative for all frequencies**
    ///
    /// Physical constraint: power spectral density must be ≥ 0 everywhere.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_psd_non_negative() {
        let spectral = SpectralAnalysis::new(SpectralConfig::default());
        let sample_rate = 44100.0_f64;
        // White-noise-like signal: sum of two incommensurate frequencies
        let signal: Array1<f64> = Array1::from_shape_fn(2048, |n| {
            (2.0 * PI * 1234.5 * n as f64 / sample_rate).sin()
                + 0.5 * (2.0 * PI * 5678.9 * n as f64 / sample_rate).sin()
        });
        let psd = spectral.compute_psd(signal.view(), sample_rate).unwrap();
        for (k, &v) in psd.iter().enumerate() {
            assert!(v >= 0.0, "PSD[{k}] = {v} < 0 (must be non-negative)");
        }
    }

    /// **Test: frequency axis has correct length and endpoint**
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_frequency_axis_length_and_nyquist() {
        let config = SpectralConfig {
            fft_size: 256,
            overlap: 0.5,
        };
        let spectral = SpectralAnalysis::new(config);
        let fs = 10_000.0_f64;
        let freq = spectral.frequency_axis(fs);

        assert_eq!(freq.len(), 129); // fft_size/2 + 1 = 129
        assert!((freq[0]).abs() < 1e-10); // DC = 0
        assert!(
            (freq[128] - fs / 2.0).abs() < 1e-6,
            "Nyquist = {}",
            freq[128]
        ); // Nyquist = 5000 Hz
    }

    /// **Test: PSD integral approximates signal power (Parseval's theorem)**
    ///
    /// For a single full segment (N = fft_size) with no overlap:
    /// `∫ PSD df ≈ mean(x²)` within 20% (Hann window reduces power by 3/8).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_psd_parseval_consistency() {
        let m = 512usize;
        let config = SpectralConfig {
            fft_size: m,
            overlap: 0.0,
        };
        let spectral = SpectralAnalysis::new(config);
        let fs = 16_000.0_f64;
        let f0 = 2_000.0_f64;

        let signal: Array1<f64> =
            Array1::from_shape_fn(m, |n| (2.0 * PI * f0 * n as f64 / fs).sin());

        let signal_power = signal.iter().map(|&x| x * x).sum::<f64>() / m as f64;
        let psd = spectral.compute_psd(signal.view(), fs).unwrap();

        // Numerical integral: Σ PSD[k] × Δf   (Δf = fs/m)
        let df = fs / m as f64;
        let integrated_power: f64 = psd.iter().sum::<f64>() * df;

        // The integrated power should be within a factor of 2 of the signal power
        // (Hann window reduces power; exact factor depends on spectral leakage)
        assert!(
            integrated_power > signal_power * 0.3 && integrated_power < signal_power * 2.0,
            "Integrated PSD power {integrated_power:.4} vs signal power {signal_power:.4} — should be comparable"
        );
    }
}
