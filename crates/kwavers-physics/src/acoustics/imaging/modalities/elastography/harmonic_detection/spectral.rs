//! Low-level spectral processing for harmonic detection (FFT, windowing, SNR)

use super::detector::HarmonicDetector;
use apollo::{fft_1d_array_into, Complex64};
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array1;

/// Caller-scoped spectral storage reused across every spatial point.
pub(super) struct SpectralWorkspace {
    window: Box<[f64]>,
    windowed: Array1<f64>,
    spectrum: Array1<Complex64>,
}

impl SpectralWorkspace {
    pub(super) fn new(sample_count: usize) -> Self {
        debug_assert!(sample_count >= 2);
        Self {
            window: (0..sample_count)
                .map(|index| hann_weight(index, sample_count))
                .collect(),
            windowed: Array1::from(vec![0.0; sample_count]),
            spectrum: Array1::from(vec![Complex64::default(); sample_count]),
        }
    }

    /// Transform one point without allocating or rebuilding its Hann weights.
    pub(super) fn transform<I>(&mut self, samples: I) -> &[Complex64]
    where
        I: ExactSizeIterator<Item = f64>,
    {
        debug_assert_eq!(samples.len(), self.window.len());
        let windowed = self
            .windowed
            .as_slice_mut()
            .expect("invariant: harmonic FFT input is C-contiguous");
        for ((output, sample), &weight) in windowed.iter_mut().zip(samples).zip(&self.window) {
            *output = sample * weight;
        }

        fft_1d_array_into(&self.windowed, &mut self.spectrum);
        let normalization = (self.window.len() as f64).sqrt();
        let independent_bin_count = self.window.len() / 2 + 1;
        let spectrum = self
            .spectrum
            .as_slice_mut()
            .expect("invariant: harmonic FFT output is C-contiguous");
        for value in &mut spectrum[..independent_bin_count] {
            *value /= normalization;
        }
        &spectrum[..independent_bin_count]
    }
}

impl HarmonicDetector {
    pub(super) fn harmonic_component(
        spectrum: &[Complex64],
        sampling_frequency: f64,
        fundamental_frequency: f64,
        harmonic_order: usize,
        sample_count: usize,
    ) -> (f64, f64, f64) {
        let bin = harmonic_order as f64 * fundamental_frequency / sampling_frequency
            * sample_count as f64;
        if !bin.is_finite() || bin < 0.0 || bin >= spectrum.len() as f64 {
            return (0.0, 0.0, 0.0);
        }

        // Truncation intentionally selects the FFT bin whose lower edge contains
        // the requested frequency, preserving the detector's established mapping.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the finite non-negative bin is range-checked against slice length"
        )]
        let index = bin as usize;
        let value = spectrum[index];
        (
            value.norm(),
            value.arg(),
            Self::compute_snr(spectrum, index),
        )
    }

    /// Compute signal-to-noise ratio at given frequency bin
    pub(crate) fn compute_snr(spectrum: &[Complex64], signal_idx: usize) -> f64 {
        let signal_power = spectrum[signal_idx].norm().powi(2);

        // Compute noise power (average of neighboring bins, excluding signal)
        let noise_radius = 10; // Use ±10 bins for noise estimation
        let mut noise_power_sum = 0.0;
        let mut noise_count = 0;

        for offset in 1..=noise_radius {
            // Left side
            if signal_idx >= offset {
                noise_power_sum += spectrum[signal_idx - offset].norm().powi(2);
                noise_count += 1;
            }

            // Right side
            if signal_idx + offset < spectrum.len() {
                noise_power_sum += spectrum[signal_idx + offset].norm().powi(2);
                noise_count += 1;
            }
        }

        let noise_power = if noise_count > 0 {
            noise_power_sum / noise_count as f64
        } else {
            1e-12 // Very small noise floor
        };

        // Convert to dB
        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            100.0 // Very high SNR if no noise detected
        }
    }
}

pub(super) fn hann_weight(index: usize, sample_count: usize) -> f64 {
    debug_assert!(sample_count >= 2);
    0.5 * (1.0 - (TWO_PI * index as f64 / (sample_count - 1) as f64).cos())
}
