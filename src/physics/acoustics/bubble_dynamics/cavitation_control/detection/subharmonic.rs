//! Subharmonic-based cavitation detection

use super::constants::{MIN_SPECTRAL_POWER, SUBHARMONIC_THRESHOLD};
use super::traits::{CavitationDetector, DetectorParameters};
use super::types::{CavitationMetrics, CavitationState, DetectionMethod};
use ndarray::{Array1, ArrayView1};
use rustfft::{num_complex::Complex, FftPlanner};

/// Subharmonic detector for stable cavitation
pub struct SubharmonicDetector {
    fundamental_freq: f64,
    sample_rate: f64,
    fft_planner: FftPlanner<f64>,
    sensitivity: f64,
}

impl std::fmt::Debug for SubharmonicDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubharmonicDetector")
            .field("fundamental_freq", &self.fundamental_freq)
            .field("sample_rate", &self.sample_rate)
            .field("fft_planner", &"<FftPlanner>")
            .field("sensitivity", &self.sensitivity)
            .finish()
    }
}

impl SubharmonicDetector {
    #[must_use]
    pub fn new(fundamental_freq: f64, sample_rate: f64) -> Self {
        Self {
            fundamental_freq,
            sample_rate,
            fft_planner: FftPlanner::new(),
            sensitivity: 1.0,
        }
    }

    /// Compute FFT and return magnitude spectrum
    fn compute_spectrum(&mut self, signal: &ArrayView1<f64>) -> Array1<f64> {
        let n = signal.len();
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(n);
        fft.process(&mut complex_signal);

        // Convert to magnitude spectrum
        Array1::from_vec(
            complex_signal
                .iter()
                .take(n / 2)
                .map(|c| c.norm() / n as f64)
                .collect(),
        )
    }

    /// Detect subharmonic components
    fn detect_subharmonic_components(&mut self, signal: &ArrayView1<f64>) -> (f64, f64) {
        let spectrum = self.compute_spectrum(signal);
        let freq_resolution = self.sample_rate / signal.len() as f64;

        // Find fundamental frequency bin
        let fundamental_bin = (self.fundamental_freq / freq_resolution) as usize;
        if fundamental_bin >= spectrum.len() {
            return (0.0, 0.0);
        }

        // Get fundamental amplitude
        let fundamental_amp = spectrum[fundamental_bin];

        // Check for f0/2 subharmonic
        let sub_bin = fundamental_bin / 2;
        let subharmonic_level = if sub_bin < spectrum.len() && fundamental_amp > MIN_SPECTRAL_POWER
        {
            spectrum[sub_bin] / fundamental_amp
        } else {
            0.0
        };

        // Check for f0/3 subharmonic (indicates stronger cavitation)
        let sub3_bin = fundamental_bin / 3;
        let sub3_level = if sub3_bin < spectrum.len() && fundamental_amp > MIN_SPECTRAL_POWER {
            spectrum[sub3_bin] / fundamental_amp
        } else {
            0.0
        };

        (subharmonic_level, sub3_level)
    }
}

impl CavitationDetector for SubharmonicDetector {
    fn detect(&mut self, signal: &ArrayView1<f64>) -> CavitationMetrics {
        let (sub2_level, sub3_level) = self.detect_subharmonic_components(signal);

        // Combine subharmonic levels
        let total_subharmonic = (sub2_level + sub3_level * 1.5).min(1.0);

        // Apply sensitivity scaling
        let scaled_level = total_subharmonic * self.sensitivity;

        // Determine cavitation state based on subharmonic levels
        let state = if scaled_level > SUBHARMONIC_THRESHOLD * 3.0 {
            CavitationState::Inertial
        } else if scaled_level > SUBHARMONIC_THRESHOLD {
            CavitationState::Stable
        } else {
            CavitationState::None
        };

        CavitationMetrics {
            state,
            subharmonic_level: scaled_level,
            ultraharmonic_level: 0.0,
            broadband_level: 0.0,
            harmonic_distortion: 0.0,
            confidence: scaled_level.min(1.0),
            // Legacy compatibility
            intensity: scaled_level.min(1.0),
            harmonic_content: 0.0,
            cavitation_dose: 0.0,
        }
    }

    fn reset(&mut self) {
        // No state to reset
    }

    fn method(&self) -> DetectionMethod {
        DetectionMethod::Subharmonic
    }

    fn update_parameters(&mut self, params: DetectorParameters) {
        self.fundamental_freq = params.fundamental_freq;
        self.sample_rate = params.sample_rate;
        self.sensitivity = params.sensitivity;
    }
}
