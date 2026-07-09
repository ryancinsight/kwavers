//! Spectral-based cavitation detection

use super::constants::{BROADBAND_THRESHOLD_DB, MIN_SPECTRAL_POWER, SPECTRAL_WINDOW_SIZE};
use super::traits::{CavitationDetector, DetectorParameters};
use super::types::{CavitationDetectionState, CavitationMetrics, DetectionMethod, HistoryBuffer};
use apollo::fft_1d_leto;
use kwavers_core::constants::numerical::TWO_PI;
use leto::application::reduction::{mean_all, std_all, sum_all};
use leto::{Array1, ArrayView1};

/// Spectral detector for cavitation using FFT analysis
pub struct SpectralDetector {
    fundamental_freq: f64,
    sample_rate: f64,
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
        let window = Self::create_hann_window(SPECTRAL_WINDOW_SIZE);

        Self {
            fundamental_freq,
            sample_rate,
            window,
            history: HistoryBuffer::new(10),
            baseline_spectrum: None,
            cumulative_dose: 0.0,
        }
    }

    /// Create Hann window for spectral analysis
    fn create_hann_window(size: usize) -> Array1<f64> {
        Array1::from_shape_fn([size], |[i]| {
            0.5 * (1.0 - (TWO_PI * i as f64 / (size - 1) as f64).cos())
        })
    }

    /// Compute power spectral density
    fn compute_psd(&mut self, signal: &ArrayView1<f64>) -> Array1<f64> {
        let n = signal.size().min(SPECTRAL_WINDOW_SIZE);

        // Apply window
        let mut windowed = leto::Array1::from_vec(
            [n],
            signal
                .iter()
                .take(n)
                .zip(self.window.iter().take(n))
                .map(|(&s, &w)| s * w)
                .collect(),
        )
        .expect("windowed length matches n");

        // Pad if necessary
        if windowed.size() < SPECTRAL_WINDOW_SIZE {
            let mut padded = leto::Array1::<f64>::zeros([SPECTRAL_WINDOW_SIZE]);
            for (i, val) in windowed.iter().enumerate() {
                padded[[i]] = *val;
            }
            windowed = padded;
        }

        let fft_input = leto::Array1::from_shape_vec(
            [SPECTRAL_WINDOW_SIZE],
            windowed.iter().copied().collect(),
        )
        .expect("windowed spectral record must match Leto FFT shape");
        let spectrum = fft_1d_leto(fft_input.view());

        // Convert to power spectral density
        let psd: Array1<f64> = spectrum
            .iter()
            .take(SPECTRAL_WINDOW_SIZE / 2)
            .map(|c| c.norm_sqr() / (SPECTRAL_WINDOW_SIZE as f64 * self.sample_rate))
            .collect::<Array1<f64>>();

        psd
    }

    /// Detect subharmonics
    fn detect_subharmonics(&self, psd: &Array1<f64>) -> f64 {
        let freq_resolution = self.sample_rate / SPECTRAL_WINDOW_SIZE as f64;
        let fundamental_bin = (self.fundamental_freq / freq_resolution) as usize;

        if fundamental_bin >= psd.size() {
            return 0.0;
        }

        // Get power around fundamental (with tolerance for frequency shift)
        let fundamental_power = if fundamental_bin > 0 && fundamental_bin < psd.size() - 1 {
            let start = fundamental_bin.saturating_sub(1);
            let end = (fundamental_bin + 1).min(psd.size() - 1);
            (start..=end)
                .map(|i| psd[[i]])
                .fold(0.0_f64, |a, b| a.max(b))
        } else {
            psd[fundamental_bin]
        };

        // Check f0/2 with tolerance
        let subharmonic_bin = fundamental_bin / 2;
        if subharmonic_bin > 0 && subharmonic_bin < psd.size() - 1 {
            let start = subharmonic_bin.saturating_sub(1);
            let end = (subharmonic_bin + 1).min(psd.size() - 1);
            let subharmonic_power = (start..=end)
                .map(|i| psd[[i]])
                .fold(0.0_f64, |a, b| a.max(b));

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

        if fundamental_bin >= psd.size() {
            return 0.0;
        }

        let fundamental_power = psd[fundamental_bin];
        let mut ultraharmonic_sum = 0.0;

        // Check 3f0/2, 5f0/2, 7f0/2
        for n in &[3, 5, 7] {
            let ultra_bin = (n * fundamental_bin) / 2;
            if ultra_bin < psd.size() {
                ultraharmonic_sum += psd[ultra_bin];
            }
        }

        ultraharmonic_sum / fundamental_power.max(MIN_SPECTRAL_POWER)
    }

    /// Detect broadband noise increase
    fn detect_broadband(&self, psd: &Array1<f64>) -> f64 {
        if let Some(baseline) = &self.baseline_spectrum {
            // Compare with baseline
            let current_energy: f64 = sum_all(&psd).unwrap_or(0.0);
            let baseline_energy: f64 = sum_all(baseline).unwrap_or(0.0);

            if baseline_energy > MIN_SPECTRAL_POWER {
                let db_increase = 10.0 * (current_energy / baseline_energy).log10();
                if db_increase > BROADBAND_THRESHOLD_DB {
                    return (db_increase / 20.0).min(1.0);
                }
            }
        }

        // Fallback: check for elevated noise floor
        let mean_power = mean_all(&psd).unwrap_or(0.0);
        let std_power = std_all(&psd, 0.0).unwrap_or(0.0);

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

        if fundamental_bin >= psd.size() {
            return 0.0;
        }

        let fundamental_power = psd[fundamental_bin];
        let mut harmonic_sum = 0.0;

        // Check harmonics up to 5th
        for n in 2..=5 {
            let harmonic_bin = n * fundamental_bin;
            if harmonic_bin < psd.size() {
                harmonic_sum += psd[harmonic_bin];
            }
        }

        harmonic_sum / fundamental_power.max(MIN_SPECTRAL_POWER)
    }

    /// Classify cavitation state based on spectral features
    fn classify_state(&self, metrics: &CavitationMetrics) -> CavitationDetectionState {
        if metrics.confidence < 0.1 {
            CavitationDetectionState::None
        } else if metrics.subharmonic_level > 0.3 || metrics.broadband_level > 0.5 {
            CavitationDetectionState::Inertial
        } else if metrics.harmonic_distortion > 0.2 {
            CavitationDetectionState::Stable
        } else {
            CavitationDetectionState::Transient
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
        let total_power: f64 = sum_all(&psd).unwrap_or(0.0);
        let confidence = if total_power > MIN_SPECTRAL_POWER {
            f64::min(1.0 - (-total_power.log10()).exp(), 1.0)
        } else {
            0.0
        };

        // Create metrics
        let mut metrics = CavitationMetrics {
            state: CavitationDetectionState::None,
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
        if metrics.state != CavitationDetectionState::None {
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
