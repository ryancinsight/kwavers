//! Spectral-based cavitation detection

use super::constants::{BROADBAND_THRESHOLD_DB, MIN_SPECTRAL_POWER, SPECTRAL_WINDOW_SIZE};
use super::traits::{CavitationDetector, DetectorParameters};
use super::types::{CavitationMetrics, CavitationState, DetectionMethod, HistoryBuffer};
use ndarray::{s, Array1, ArrayView1};
use rustfft::{num_complex::Complex, FftPlanner};

/// Spectral detector for cavitation using FFT analysis
pub struct SpectralDetector {
    fundamental_freq: f64,
    sample_rate: f64,
    fft_planner: FftPlanner<f64>,
    window: Array1<f64>,
    history: HistoryBuffer<CavitationMetrics>,
    baseline_spectrum: Option<Array1<f64>>,
    cumulative_dose: f64,
}

impl std::fmt::Debug for SpectralDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralDetector")
            .field("fundamental_freq", &self.fundamental_freq)
            .field("sample_rate", &self.sample_rate)
            .field("fft_planner", &"<FftPlanner>")
            .field("window_len", &self.window.len())
            .field("history_len", &self.history.len())
            .field("has_baseline", &self.baseline_spectrum.is_some())
            .field("cumulative_dose", &self.cumulative_dose)
            .finish()
    }
}

impl SpectralDetector {
    #[must_use]
    pub fn new(fundamental_freq: f64, sample_rate: f64) -> Self {
        let fft_planner = FftPlanner::new();
        let window = Self::create_hann_window(SPECTRAL_WINDOW_SIZE);

        Self {
            fundamental_freq,
            sample_rate,
            fft_planner,
            window,
            history: HistoryBuffer::new(10),
            baseline_spectrum: None,
            cumulative_dose: 0.0,
        }
    }

    /// Create Hann window for spectral analysis
    fn create_hann_window(size: usize) -> Array1<f64> {
        Array1::from_shape_fn(size, |i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())
        })
    }

    /// Compute power spectral density
    fn compute_psd(&mut self, signal: &ArrayView1<f64>) -> Array1<f64> {
        let n = signal.len().min(SPECTRAL_WINDOW_SIZE);

        // Apply window
        let mut windowed: Vec<Complex<f64>> = signal
            .iter()
            .take(n)
            .zip(self.window.iter().take(n))
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Pad if necessary
        windowed.resize(SPECTRAL_WINDOW_SIZE, Complex::new(0.0, 0.0));

        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(SPECTRAL_WINDOW_SIZE);
        fft.process(&mut windowed);

        // Convert to power spectral density
        let psd: Array1<f64> = windowed
            .iter()
            .take(SPECTRAL_WINDOW_SIZE / 2)
            .map(|c| (c.norm_sqr() / (SPECTRAL_WINDOW_SIZE as f64 * self.sample_rate)))
            .collect();

        psd
    }

    /// Detect subharmonics
    fn detect_subharmonics(&self, psd: &Array1<f64>) -> f64 {
        let freq_resolution = self.sample_rate / SPECTRAL_WINDOW_SIZE as f64;
        let fundamental_bin = (self.fundamental_freq / freq_resolution) as usize;

        if fundamental_bin >= psd.len() {
            return 0.0;
        }

        // Get power around fundamental (with tolerance for frequency shift)
        let fundamental_power = if fundamental_bin > 0 && fundamental_bin < psd.len() - 1 {
            let start = fundamental_bin.saturating_sub(1);
            let end = (fundamental_bin + 1).min(psd.len() - 1);
            psd.slice(s![start..=end])
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b))
        } else {
            psd[fundamental_bin]
        };

        // Check f0/2 with tolerance
        let subharmonic_bin = fundamental_bin / 2;
        if subharmonic_bin > 0 && subharmonic_bin < psd.len() - 1 {
            let start = subharmonic_bin.saturating_sub(1);
            let end = (subharmonic_bin + 1).min(psd.len() - 1);
            let subharmonic_power = psd
                .slice(s![start..=end])
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b));

            if subharmonic_power > MIN_SPECTRAL_POWER {
                return (subharmonic_power / fundamental_power.max(MIN_SPECTRAL_POWER)).min(1.0);
            }
        }

        0.0
    }

    /// Detect ultraharmonics (n*f0/2 where n is odd)
    fn detect_ultraharmonics(&self, psd: &Array1<f64>) -> f64 {
        let freq_resolution = self.sample_rate / SPECTRAL_WINDOW_SIZE as f64;
        let fundamental_bin = (self.fundamental_freq / freq_resolution) as usize;

        if fundamental_bin >= psd.len() {
            return 0.0;
        }

        let fundamental_power = psd[fundamental_bin];
        let mut ultraharmonic_sum = 0.0;

        // Check 3f0/2, 5f0/2, 7f0/2
        for n in &[3, 5, 7] {
            let ultra_bin = (n * fundamental_bin) / 2;
            if ultra_bin < psd.len() {
                ultraharmonic_sum += psd[ultra_bin];
            }
        }

        ultraharmonic_sum / fundamental_power.max(MIN_SPECTRAL_POWER)
    }

    /// Detect broadband noise increase
    fn detect_broadband(&self, psd: &Array1<f64>) -> f64 {
        if let Some(baseline) = &self.baseline_spectrum {
            // Compare with baseline
            let current_energy: f64 = psd.sum();
            let baseline_energy: f64 = baseline.sum();

            if baseline_energy > MIN_SPECTRAL_POWER {
                let db_increase = 10.0 * (current_energy / baseline_energy).log10();
                if db_increase > BROADBAND_THRESHOLD_DB {
                    return (db_increase / 20.0).min(1.0);
                }
            }
        }

        // Fallback: check for elevated noise floor
        let mean_power = psd.mean().unwrap_or(0.0);
        let std_power = psd.std(0.0);

        if mean_power > MIN_SPECTRAL_POWER {
            (std_power / mean_power).min(1.0)
        } else {
            0.0
        }
    }

    /// Detect harmonic distortion
    fn detect_harmonics(&self, psd: &Array1<f64>) -> f64 {
        let freq_resolution = self.sample_rate / SPECTRAL_WINDOW_SIZE as f64;
        let fundamental_bin = (self.fundamental_freq / freq_resolution) as usize;

        if fundamental_bin >= psd.len() {
            return 0.0;
        }

        let fundamental_power = psd[fundamental_bin];
        let mut harmonic_sum = 0.0;

        // Check harmonics up to 5th
        for n in 2..=5 {
            let harmonic_bin = n * fundamental_bin;
            if harmonic_bin < psd.len() {
                harmonic_sum += psd[harmonic_bin];
            }
        }

        harmonic_sum / fundamental_power.max(MIN_SPECTRAL_POWER)
    }

    /// Classify cavitation state based on spectral features
    fn classify_state(&self, metrics: &CavitationMetrics) -> CavitationState {
        if metrics.confidence < 0.1 {
            CavitationState::None
        } else if metrics.subharmonic_level > 0.3 || metrics.broadband_level > 0.5 {
            CavitationState::Inertial
        } else if metrics.harmonic_distortion > 0.2 {
            CavitationState::Stable
        } else {
            CavitationState::Transient
        }
    }

    /// Update baseline spectrum for adaptive detection
    pub fn update_baseline(&mut self, signal: &ArrayView1<f64>) {
        self.baseline_spectrum = Some(self.compute_psd(signal));
    }
}

impl CavitationDetector for SpectralDetector {
    fn detect(&mut self, signal: &ArrayView1<f64>) -> CavitationMetrics {
        // Compute power spectral density
        let psd = self.compute_psd(signal);

        // Detect various spectral features
        let subharmonic_level = self.detect_subharmonics(&psd);
        let ultraharmonic_level = self.detect_ultraharmonics(&psd);
        let broadband_level = self.detect_broadband(&psd);
        let harmonic_distortion = self.detect_harmonics(&psd);

        // Calculate confidence based on signal quality
        let total_power: f64 = psd.sum();
        let confidence = if total_power > MIN_SPECTRAL_POWER {
            (1.0 - (-total_power.log10()).exp()).min(1.0)
        } else {
            0.0
        };

        // Create metrics
        let mut metrics = CavitationMetrics {
            state: CavitationState::None,
            subharmonic_level,
            ultraharmonic_level,
            broadband_level,
            harmonic_distortion,
            confidence,
            // Legacy compatibility
            intensity: confidence,
            harmonic_content: harmonic_distortion,
            cavitation_dose: self.cumulative_dose,
        };

        // Classify state
        metrics.state = self.classify_state(&metrics);

        // Update history for temporal analysis
        self.history.push(metrics.clone());

        // Update cumulative dose if cavitation detected
        if metrics.state != CavitationState::None {
            self.cumulative_dose += metrics.confidence;
        }

        metrics
    }

    fn reset(&mut self) {
        self.history = HistoryBuffer::new(10);
        self.baseline_spectrum = None;
        self.cumulative_dose = 0.0;
    }

    fn method(&self) -> DetectionMethod {
        DetectionMethod::Combined
    }

    fn update_parameters(&mut self, params: DetectorParameters) {
        self.fundamental_freq = params.fundamental_freq;
        self.sample_rate = params.sample_rate;
        // Recreate window if size changes
        self.window = Self::create_hann_window(SPECTRAL_WINDOW_SIZE);
    }
}
