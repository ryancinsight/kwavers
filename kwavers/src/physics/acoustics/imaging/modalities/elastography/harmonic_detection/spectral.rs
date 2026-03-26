//! Low-level spectral processing for harmonic detection (FFT, windowing, SNR)

use super::detector::HarmonicDetector;
use super::types::PointHarmonics;
use crate::core::error::KwaversResult;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

impl HarmonicDetector {
    /// Analyze harmonics at a single spatial point
    pub(crate) fn analyze_single_point(
        &self,
        time_series: &[f64],
        sampling_frequency: f64,
    ) -> KwaversResult<PointHarmonics> {
        // Apply windowing and FFT
        let windowed = self.apply_window(time_series);
        let spectrum = self.compute_fft(&windowed)?;

        // Find fundamental frequency peak
        let _fundamental_idx = self.find_fundamental_peak(&spectrum, sampling_frequency)?;

        // Extract harmonic components
        let mut harmonic_magnitudes = Vec::new();
        let mut harmonic_phases = Vec::new();
        let mut harmonic_snrs = Vec::new();

        for harmonic_order in 1..=self.config.n_harmonics {
            let harmonic_freq = harmonic_order as f64 * self.config.fundamental_frequency;
            let harmonic_idx =
                (harmonic_freq / sampling_frequency * spectrum.len() as f64) as usize;

            if harmonic_idx < spectrum.len() {
                let magnitude = spectrum[harmonic_idx].norm();
                let phase = spectrum[harmonic_idx].arg();

                // Compute SNR
                let snr = self.compute_snr(&spectrum, harmonic_idx);

                harmonic_magnitudes.push(magnitude);
                harmonic_phases.push(phase);
                harmonic_snrs.push(snr);
            } else {
                harmonic_magnitudes.push(0.0);
                harmonic_phases.push(0.0);
                harmonic_snrs.push(0.0);
            }
        }

        // Extract fundamental (first harmonic)
        let fundamental_magnitude = harmonic_magnitudes[0];
        let fundamental_phase = harmonic_phases[0];

        // Remove fundamental from harmonic list
        harmonic_magnitudes.remove(0);
        harmonic_phases.remove(0);
        harmonic_snrs.remove(0);

        Ok(PointHarmonics {
            fundamental_magnitude,
            fundamental_phase,
            harmonic_magnitudes,
            harmonic_phases,
            harmonic_snrs,
        })
    }

    /// Apply window function to time series
    pub(crate) fn apply_window(&self, time_series: &[f64]) -> Vec<f64> {
        let n = time_series.len();
        let mut windowed = Vec::with_capacity(n);

        // Hann window
        for (i, &val) in time_series.iter().enumerate().take(n) {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            windowed.push(val * window);
        }

        windowed
    }

    /// Compute FFT of time series
    pub(crate) fn compute_fft(&self, time_series: &[f64]) -> KwaversResult<Vec<Complex<f64>>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(time_series.len());

        let mut buffer: Vec<Complex<f64>> =
            time_series.iter().map(|&x| Complex::new(x, 0.0)).collect();

        fft.process(&mut buffer);

        // Normalize
        let norm_factor = (time_series.len() as f64).sqrt();
        for val in &mut buffer {
            *val /= norm_factor;
        }

        Ok(buffer)
    }

    /// Find fundamental frequency peak in spectrum
    pub(crate) fn find_fundamental_peak(
        &self,
        spectrum: &[Complex<f64>],
        sampling_frequency: f64,
    ) -> KwaversResult<usize> {
        let df = sampling_frequency / spectrum.len() as f64;
        let expected_idx = (self.config.fundamental_frequency / df) as usize;

        // Search around expected fundamental frequency
        let search_radius = 5; // ±5 bins
        let start_idx = expected_idx.saturating_sub(search_radius);
        let end_idx = (expected_idx + search_radius).min(spectrum.len() - 1);

        let mut max_magnitude = 0.0;
        let mut peak_idx = expected_idx;

        for (idx, val) in spectrum
            .iter()
            .enumerate()
            .take(end_idx + 1)
            .skip(start_idx)
        {
            let magnitude = val.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_idx = idx;
            }
        }

        Ok(peak_idx)
    }

    /// Compute signal-to-noise ratio at given frequency bin
    pub(crate) fn compute_snr(&self, spectrum: &[Complex<f64>], signal_idx: usize) -> f64 {
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
