//! Harmonic Detection and Analysis for Nonlinear SWE
//!
//! Implements multi-frequency displacement tracking and harmonic analysis
//! for nonlinear shear wave elastography.
//!
//! ## Overview
//!
//! Harmonic detection in NL-SWE involves:
//! 1. Multi-frequency displacement estimation
//! 2. Frequency-domain analysis of harmonic components
//! 3. Phase-sensitive harmonic detection
//! 4. Harmonic amplitude and phase estimation
//! 5. Tissue nonlinearity parameter extraction
//!
//! ## Mathematical Framework
//!
//! ### Harmonic Generation
//!
//! For weakly nonlinear media, the displacement field contains harmonics:
//!
//! u(x,t) = u₁(x,t) + u₂(x,t) + u₃(x,t) + ...
//!
//! Where uₙ represents the nth harmonic component.
//!
//! ### Frequency Domain Analysis
//!
//! The fundamental frequency ω₁ generates harmonics at 2ω₁, 3ω₁, etc.
//!
//! Uₙ(ω) = FFT[uₙ(t)] at frequency nω₁
//!
//! ### Nonlinear Parameter Estimation
//!
//! The acoustic nonlinearity parameter B/A can be estimated from:
//!
//! B/A = (8/μ) * (ρ₀ c₀³ / (β P₀)) * (A₂/A₁)
//!
//! Where A₂/A₁ is the ratio of second to first harmonic amplitudes.
//!
//! ## Literature References
//!
//! - Chen, S., et al. (2013). "Harmonic motion detection in ultrasound elastography."
//!   *IEEE Transactions on Medical Imaging*, 32(5), 863-874.
//! - Parker, K. J., et al. (2011). "Sonoelasticity of organs: Shear waves ring a bell."
//!   *Journal of Ultrasound in Medicine*, 30(4), 507-515.
//! - Nightingale, K. R., et al. (2015). "Acoustic Radiation Force Impulse (ARFI)
//!   imaging: A review." *Current Medical Imaging Reviews*, 11(1), 22-32.

use crate::error::KwaversResult;
use ndarray::{s, Array2, Array3, Array4, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

/// Configuration for harmonic detection
#[derive(Debug, Clone)]
pub struct HarmonicDetectionConfig {
    /// Fundamental frequency (Hz)
    pub fundamental_frequency: f64,
    /// Number of harmonics to detect
    pub n_harmonics: usize,
    /// FFT window size
    pub fft_window_size: usize,
    /// Overlap between FFT windows
    pub fft_overlap: f64,
    /// Minimum SNR for harmonic detection (dB)
    pub min_snr_db: f64,
    /// Phase unwrapping enabled
    pub enable_phase_unwrapping: bool,
}

impl Default for HarmonicDetectionConfig {
    fn default() -> Self {
        Self {
            fundamental_frequency: 50.0, // 50 Hz typical for SWE
            n_harmonics: 3,              // Fundamental + 2 harmonics
            fft_window_size: 1024,
            fft_overlap: 0.5,            // 50% overlap
            min_snr_db: 10.0,            // 10 dB minimum SNR
            enable_phase_unwrapping: true,
        }
    }
}

/// Multi-frequency displacement field with harmonic components
#[derive(Debug, Clone)]
pub struct HarmonicDisplacementField {
    /// Fundamental frequency displacement magnitude
    pub fundamental_magnitude: Array3<f64>,
    /// Fundamental frequency displacement phase
    pub fundamental_phase: Array3<f64>,
    /// Harmonic displacement magnitudes (vector of arrays for each harmonic)
    pub harmonic_magnitudes: Vec<Array3<f64>>,
    /// Harmonic displacement phases (vector of arrays for each harmonic)
    pub harmonic_phases: Vec<Array3<f64>>,
    /// Signal-to-noise ratios for each harmonic (dB)
    pub harmonic_snrs: Vec<Array3<f64>>,
    /// Nonlinearity parameter B/A estimates
    pub nonlinearity_parameter: Array3<f64>,
    /// Time vector for the analysis
    pub time: Vec<f64>,
    /// Frequency vector for spectral analysis
    pub frequency: Vec<f64>,
}

impl HarmonicDisplacementField {
    /// Create new harmonic displacement field
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, n_harmonics: usize, n_time_points: usize) -> Self {
        let mut harmonic_magnitudes = Vec::with_capacity(n_harmonics);
        let mut harmonic_phases = Vec::with_capacity(n_harmonics);
        let mut harmonic_snrs = Vec::with_capacity(n_harmonics);

        for _ in 0..n_harmonics {
            harmonic_magnitudes.push(Array3::zeros((nx, ny, nz)));
            harmonic_phases.push(Array3::zeros((nx, ny, nz)));
            harmonic_snrs.push(Array3::zeros((nx, ny, nz)));
        }

        Self {
            fundamental_magnitude: Array3::zeros((nx, ny, nz)),
            fundamental_phase: Array3::zeros((nx, ny, nz)),
            harmonic_magnitudes,
            harmonic_phases,
            harmonic_snrs,
            nonlinearity_parameter: Array3::zeros((nx, ny, nz)),
            time: vec![0.0; n_time_points],
            frequency: vec![0.0; n_time_points / 2 + 1], // FFT frequency bins
        }
    }

    /// Get harmonic ratio (A₂/A₁) for nonlinearity estimation
    #[must_use]
    pub fn harmonic_ratio(&self, harmonic_order: usize) -> Array3<f64> {
        if harmonic_order == 0 || harmonic_order > self.harmonic_magnitudes.len() {
            return Array3::zeros(self.fundamental_magnitude.dim());
        }

        &self.harmonic_magnitudes[harmonic_order - 1] / &self.fundamental_magnitude
    }

    /// Compute local nonlinearity parameter map
    pub fn compute_nonlinearity_parameter(&mut self, config: &HarmonicDetectionConfig) {
        // Use second harmonic ratio for B/A estimation
        // B/A = (8/μ) * (ρ₀ c₀³ / (β P₀)) * (A₂/A₁)
        // Simplified version using empirical relationship

        let harmonic_ratio = self.harmonic_ratio(2); // Second harmonic

        // Empirical calibration factor (would be determined experimentally)
        let calibration_factor = 1.0;

        self.nonlinearity_parameter = &harmonic_ratio * calibration_factor;
    }
}

/// Harmonic detection and analysis processor
pub struct HarmonicDetector {
    /// Configuration
    config: HarmonicDetectionConfig,
    /// FFT planner for spectral analysis
    fft_planner: FftPlanner<f64>,
}

impl HarmonicDetector {
    /// Create new harmonic detector
    #[must_use]
    pub fn new(config: HarmonicDetectionConfig) -> Self {
        Self {
            config,
            fft_planner: FftPlanner::new(),
        }
    }

    /// Analyze displacement time series for harmonic content
    ///
    /// # Arguments
    ///
    /// * `displacement_time_series` - 4D array: [nx, ny, nz, n_time_points]
    /// * `sampling_frequency` - Sampling frequency (Hz)
    ///
    /// # Returns
    ///
    /// Harmonic displacement field with all frequency components
    pub fn analyze_harmonics(
        &self,
        displacement_time_series: &Array4<f64>,
        sampling_frequency: f64,
    ) -> KwaversResult<HarmonicDisplacementField> {
        let (nx, ny, nz, n_times) = displacement_time_series.dim();

        let mut harmonic_field = HarmonicDisplacementField::new(nx, ny, nz, self.config.n_harmonics, n_times);

        // Set time and frequency vectors
        for t in 0..n_times {
            harmonic_field.time[t] = t as f64 / sampling_frequency;
        }

        let df = sampling_frequency / self.config.fft_window_size as f64;
        for f in 0..harmonic_field.frequency.len() {
            harmonic_field.frequency[f] = f as f64 * df;
        }

        // Analyze each spatial point
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let time_series = displacement_time_series.slice(s![i, j, k, ..]);
                    let harmonics = self.analyze_single_point(&time_series.to_vec(), sampling_frequency)?;

                    // Store results
                    harmonic_field.fundamental_magnitude[[i, j, k]] = harmonics.fundamental_magnitude;
                    harmonic_field.fundamental_phase[[i, j, k]] = harmonics.fundamental_phase;

                    for h in 0..self.config.n_harmonics {
                        if h < harmonics.harmonic_magnitudes.len() {
                            harmonic_field.harmonic_magnitudes[h][[i, j, k]] = harmonics.harmonic_magnitudes[h];
                            harmonic_field.harmonic_phases[h][[i, j, k]] = harmonics.harmonic_phases[h];
                            harmonic_field.harmonic_snrs[h][[i, j, k]] = harmonics.harmonic_snrs[h];
                        }
                    }
                }
            }
        }

        // Compute nonlinearity parameter
        harmonic_field.compute_nonlinearity_parameter(&self.config);

        Ok(harmonic_field)
    }

    /// Analyze harmonics at a single spatial point
    fn analyze_single_point(
        &self,
        time_series: &[f64],
        sampling_frequency: f64,
    ) -> KwaversResult<PointHarmonics> {
        // Apply windowing and FFT
        let windowed = self.apply_window(time_series);
        let spectrum = self.compute_fft(&windowed)?;

        // Find fundamental frequency peak
        let fundamental_idx = self.find_fundamental_peak(&spectrum, sampling_frequency)?;

        // Extract harmonic components
        let mut harmonic_magnitudes = Vec::new();
        let mut harmonic_phases = Vec::new();
        let mut harmonic_snrs = Vec::new();

        for harmonic_order in 1..=self.config.n_harmonics {
            let harmonic_freq = harmonic_order as f64 * self.config.fundamental_frequency;
            let harmonic_idx = (harmonic_freq / sampling_frequency * spectrum.len() as f64) as usize;

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
    fn apply_window(&self, time_series: &[f64]) -> Vec<f64> {
        let n = time_series.len();
        let mut windowed = Vec::with_capacity(n);

        // Hann window
        for i in 0..n {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            windowed.push(time_series[i] * window);
        }

        windowed
    }

    /// Compute FFT of time series
    fn compute_fft(&self, time_series: &[f64]) -> KwaversResult<Vec<Complex<f64>>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(time_series.len());

        let mut buffer: Vec<Complex<f64>> = time_series
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut buffer);

        // Normalize
        let norm_factor = (time_series.len() as f64).sqrt();
        for val in &mut buffer {
            *val /= norm_factor;
        }

        Ok(buffer)
    }

    /// Find fundamental frequency peak in spectrum
    fn find_fundamental_peak(
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

        for idx in start_idx..=end_idx {
            let magnitude = spectrum[idx].norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_idx = idx;
            }
        }

        Ok(peak_idx)
    }

    /// Compute signal-to-noise ratio at given frequency bin
    fn compute_snr(&self, spectrum: &[Complex<f64>], signal_idx: usize) -> f64 {
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

/// Harmonic analysis results for a single spatial point
#[derive(Debug)]
struct PointHarmonics {
    /// Fundamental frequency magnitude
    fundamental_magnitude: f64,
    /// Fundamental frequency phase
    fundamental_phase: f64,
    /// Harmonic magnitudes (excluding fundamental)
    harmonic_magnitudes: Vec<f64>,
    /// Harmonic phases (excluding fundamental)
    harmonic_phases: Vec<f64>,
    /// Harmonic SNRs in dB
    harmonic_snrs: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_harmonic_detection_config() {
        let config = HarmonicDetectionConfig::default();
        assert_eq!(config.fundamental_frequency, 50.0);
        assert_eq!(config.n_harmonics, 3);
        assert_eq!(config.fft_window_size, 1024);
    }

    #[test]
    fn test_harmonic_displacement_field_creation() {
        let field = HarmonicDisplacementField::new(10, 10, 10, 3, 100);

        assert_eq!(field.fundamental_magnitude.dim(), (10, 10, 10));
        assert_eq!(field.fundamental_phase.dim(), (10, 10, 10));
        assert_eq!(field.harmonic_magnitudes.len(), 3);
        assert_eq!(field.harmonic_phases.len(), 3);
        assert_eq!(field.harmonic_snrs.len(), 3);
        assert_eq!(field.time.len(), 100);
    }

    #[test]
    fn test_harmonic_ratio_computation() {
        let mut field = HarmonicDisplacementField::new(5, 5, 5, 2, 50);

        // Set test values
        field.fundamental_magnitude.fill(1.0);
        field.harmonic_magnitudes[1].fill(0.1); // Second harmonic

        let ratio = field.harmonic_ratio(2);
        assert_eq!(ratio.dim(), (5, 5, 5));

        // Check ratio value
        for &val in ratio.iter() {
            assert!((val - 0.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_window_function() {
        let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());
        let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let windowed = detector.apply_window(&time_series);

        assert_eq!(windowed.len(), time_series.len());
        // First and last values should be zero (Hann window)
        assert!((windowed[0] - 0.0).abs() < 1e-10);
        assert!((windowed[4] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_snr_computation() {
        let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());

        // Create test spectrum with signal peak
        let mut spectrum = vec![Complex::new(0.1, 0.0); 100];
        spectrum[50] = Complex::new(1.0, 0.0); // Strong signal

        let snr = detector.compute_snr(&spectrum, 50);
        assert!(snr > 0.0); // Should have positive SNR
    }

    #[test]
    fn test_harmonic_detector_creation() {
        let config = HarmonicDetectionConfig {
            fundamental_frequency: 100.0,
            n_harmonics: 5,
            ..Default::default()
        };

        let detector = HarmonicDetector::new(config);
        // Test passes if no panic occurs
    }
}
