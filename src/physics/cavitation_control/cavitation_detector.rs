//! Cavitation Detection Module
//! 
//! Implements various methods for detecting and quantifying cavitation activity
//! using acoustic emissions and spectral analysis.
//! 
//! References:
//! - Gyöngy & Coussios (2010): "Passive cavitation mapping for localization and tracking"
//! - Chen et al. (2003): "Inertial cavitation dose and hemolysis produced in vitro"
//! - Mast et al. (2008): "Acoustic emissions during 3.1 MHz ultrasound bulk ablation"

use ndarray::{Array1, ArrayView1};
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::VecDeque;

// Detection constants
/// Subharmonic detection threshold (relative to fundamental)
const SUBHARMONIC_THRESHOLD: f64 = 0.1;

/// Broadband noise floor increase for cavitation detection (dB)
const BROADBAND_THRESHOLD_DB: f64 = 6.0;

/// Harmonic detection threshold (relative to fundamental)
const HARMONIC_THRESHOLD: f64 = 0.05;

/// Minimum spectral power for valid detection
const MIN_SPECTRAL_POWER: f64 = 1e-6;

/// Window size for spectral analysis
const SPECTRAL_WINDOW_SIZE: usize = 1024;

/// Overlap ratio for spectral windows
const WINDOW_OVERLAP_RATIO: f64 = 0.5;

/// Detection methods for cavitation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionMethod {
    Subharmonic,     // f0/2, f0/3, etc.
    Ultraharmonic,   // 3f0/2, 5f0/2, etc.
    Broadband,       // Increased broadband noise
    Harmonic,        // 2f0, 3f0, etc.
    Combined,        // Combination of methods
}

/// Cavitation state classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CavitationState {
    None,
    Stable,      // Stable cavitation (non-inertial)
    Inertial,    // Inertial cavitation (violent collapse)
    Transient,   // Transitioning between states
}

/// Cavitation metrics
#[derive(Debug, Clone)]
pub struct CavitationMetrics {
    pub state: CavitationState,
    pub intensity: f64,           // 0-1 normalized
    pub subharmonic_level: f64,   // Subharmonic amplitude
    pub broadband_level: f64,     // Broadband noise level
    pub harmonic_content: f64,    // Harmonic distortion
    pub cavitation_dose: f64,     // Cumulative dose
    pub confidence: f64,           // Detection confidence 0-1
}

impl Default for CavitationMetrics {
    fn default() -> Self {
        Self {
            state: CavitationState::None,
            intensity: 0.0,
            subharmonic_level: 0.0,
            broadband_level: 0.0,
            harmonic_content: 0.0,
            cavitation_dose: 0.0,
            confidence: 0.0,
        }
    }
}

/// Base cavitation detector trait
pub trait CavitationDetector {
    fn detect(&mut self, signal: &ArrayView1<f64>) -> CavitationMetrics;
    fn reset(&mut self);
    fn set_fundamental_frequency(&mut self, frequency: f64);
}

/// Spectral-based cavitation detector
pub struct SpectralDetector {
    fundamental_freq: f64,
    sample_rate: f64,
    fft_planner: FftPlanner<f64>,
    window: Array1<f64>,
    history: VecDeque<CavitationMetrics>,
    baseline_spectrum: Option<Array1<f64>>,
    cumulative_dose: f64,
}

impl SpectralDetector {
    pub fn new(fundamental_freq: f64, sample_rate: f64) -> Self {
        let fft_planner = FftPlanner::new();
        let window = Self::create_hann_window(SPECTRAL_WINDOW_SIZE);
        
        Self {
            fundamental_freq,
            sample_rate,
            fft_planner,
            window,
            history: VecDeque::with_capacity(10),
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
        let mut windowed: Vec<Complex<f64>> = signal.iter()
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
        let psd: Array1<f64> = windowed.iter()
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
        
        let fundamental_power = psd[fundamental_bin];
        
        // Check f0/2
        let subharmonic_bin = fundamental_bin / 2;
        if subharmonic_bin > 0 && subharmonic_bin < psd.len() {
            let subharmonic_power = psd[subharmonic_bin];
            if subharmonic_power > SUBHARMONIC_THRESHOLD * fundamental_power {
                return subharmonic_power / fundamental_power.max(MIN_SPECTRAL_POWER);
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
        for n in [3, 5, 7].iter() {
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
                    return (db_increase / 20.0).min(1.0); // Normalize to 0-1
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
    
    /// Classify cavitation state based on spectral features
    fn classify_state(&self, metrics: &CavitationMetrics) -> CavitationState {
        if metrics.intensity < 0.1 {
            CavitationState::None
        } else if metrics.subharmonic_level > 0.3 || metrics.broadband_level > 0.5 {
            CavitationState::Inertial
        } else if metrics.harmonic_content > 0.2 {
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
        let psd = self.compute_psd(signal);
        
        // Detect various cavitation signatures
        let subharmonic_level = self.detect_subharmonics(&psd);
        let ultraharmonic_level = self.detect_ultraharmonics(&psd);
        let broadband_level = self.detect_broadband(&psd);
        
        // Calculate harmonic content
        let freq_resolution = self.sample_rate / SPECTRAL_WINDOW_SIZE as f64;
        let fundamental_bin = (self.fundamental_freq / freq_resolution) as usize;
        let mut harmonic_content = 0.0;
        
        if fundamental_bin < psd.len() {
            let fundamental_power = psd[fundamental_bin];
            for n in 2..=5 {
                let harmonic_bin = n * fundamental_bin;
                if harmonic_bin < psd.len() {
                    harmonic_content += psd[harmonic_bin] / fundamental_power.max(MIN_SPECTRAL_POWER);
                }
            }
        }
        
        // Calculate overall intensity
        let intensity = (subharmonic_level * 0.4 + 
                        ultraharmonic_level * 0.3 + 
                        broadband_level * 0.3).min(1.0);
        
        // Calculate confidence based on signal strength
        let signal_power: f64 = psd.sum();
        let confidence = (signal_power / MIN_SPECTRAL_POWER).log10().max(0.0).min(1.0) / 6.0;
        
        // Update cumulative dose
        self.cumulative_dose += intensity * (1.0 / self.sample_rate);
        
        let mut metrics = CavitationMetrics {
            state: CavitationState::None,
            intensity,
            subharmonic_level,
            broadband_level,
            harmonic_content,
            cavitation_dose: self.cumulative_dose,
            confidence,
        };
        
        metrics.state = self.classify_state(&metrics);
        
        // Store in history
        self.history.push_back(metrics.clone());
        if self.history.len() > 10 {
            self.history.pop_front();
        }
        
        metrics
    }
    
    fn reset(&mut self) {
        self.history.clear();
        self.cumulative_dose = 0.0;
        self.baseline_spectrum = None;
    }
    
    fn set_fundamental_frequency(&mut self, frequency: f64) {
        self.fundamental_freq = frequency;
    }
}

/// Broadband emissions detector
pub struct BroadbandDetector {
    sample_rate: f64,
    noise_floor: f64,
    history: VecDeque<f64>,
    threshold_multiplier: f64,
}

impl BroadbandDetector {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            noise_floor: 0.0,
            history: VecDeque::with_capacity(100),
            threshold_multiplier: 2.0,
        }
    }
    
    /// Estimate noise floor from signal
    fn estimate_noise_floor(&mut self, signal: &ArrayView1<f64>) {
        // Use median absolute deviation for robust noise estimation
        let mut sorted: Vec<f64> = signal.iter().map(|&x| x.abs()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        
        self.noise_floor = median * 1.4826; // Scale factor for Gaussian noise
    }
    
    pub fn detect(&mut self, signal: &ArrayView1<f64>) -> f64 {
        if self.noise_floor == 0.0 {
            self.estimate_noise_floor(signal);
        }
        
        // Calculate RMS of high-frequency components
        let mut high_freq_energy = 0.0;
        for window in signal.windows(3) {
            let high_pass = window[1] - 0.5 * (window[0] + window[2]);
            high_freq_energy += high_pass * high_pass;
        }
        
        let rms = (high_freq_energy / signal.len() as f64).sqrt();
        
        // Detect if above threshold
        let detection = if rms > self.noise_floor * self.threshold_multiplier {
            (rms / self.noise_floor - self.threshold_multiplier) / self.threshold_multiplier
        } else {
            0.0
        };
        
        detection.min(1.0)
    }
}

/// Subharmonic-specific detector
pub struct SubharmonicDetector {
    fundamental_freq: f64,
    sample_rate: f64,
    filter_coeffs: Vec<f64>,
    state: VecDeque<f64>,
}

impl SubharmonicDetector {
    pub fn new(fundamental_freq: f64, sample_rate: f64) -> Self {
        // Design bandpass filter for f0/2
        let filter_coeffs = Self::design_bandpass_filter(
            fundamental_freq / 2.0,
            fundamental_freq / 10.0, // Bandwidth
            sample_rate
        );
        
        let filter_len = filter_coeffs.len();
        Self {
            fundamental_freq,
            sample_rate,
            filter_coeffs,
            state: VecDeque::with_capacity(filter_len),
        }
    }
    
    /// Design simple Butterworth bandpass filter
    fn design_bandpass_filter(center_freq: f64, bandwidth: f64, sample_rate: f64) -> Vec<f64> {
        let omega = 2.0 * std::f64::consts::PI * center_freq / sample_rate;
        let bw = 2.0 * std::f64::consts::PI * bandwidth / sample_rate;
        
        // Simple 2nd order bandpass
        let q = center_freq / bandwidth;
        let alpha = bw.sin() / (2.0 * q);
        
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * omega.cos();
        let a2 = 1.0 - alpha;
        
        vec![b0/a0, b1/a0, b2/a0, -a1/a0, -a2/a0]
    }
    
    pub fn detect(&mut self, signal: &ArrayView1<f64>) -> f64 {
        let mut filtered_energy = 0.0;
        let mut total_energy = 0.0;
        
        for &sample in signal.iter() {
            // Apply filter
            self.state.push_front(sample);
            if self.state.len() > self.filter_coeffs.len() {
                self.state.pop_back();
            }
            
            let mut filtered = 0.0;
            for (i, &coeff) in self.filter_coeffs.iter().enumerate() {
                if i < self.state.len() {
                    filtered += coeff * self.state[i];
                }
            }
            
            filtered_energy += filtered * filtered;
            total_energy += sample * sample;
        }
        
        if total_energy > MIN_SPECTRAL_POWER {
            (filtered_energy / total_energy).sqrt().min(1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_spectral_detector() {
        let sample_rate = 44100.0;
        let fundamental = 1000.0;
        let mut detector = SpectralDetector::new(fundamental, sample_rate);
        
        // Create test signal with fundamental and subharmonic
        let t: Array1<f64> = Array1::linspace(0.0, 0.1, 4410);
        let signal: Array1<f64> = t.mapv(|t| {
            (2.0 * std::f64::consts::PI * fundamental * t).sin() +
            0.3 * (2.0 * std::f64::consts::PI * fundamental / 2.0 * t).sin()
        });
        
        let metrics = detector.detect(&signal.view());
        
        // Should detect subharmonic
        assert!(metrics.subharmonic_level > 0.0);
    }
    
    #[test]
    fn test_cavitation_state_classification() {
        let sample_rate = 44100.0;
        let detector = SpectralDetector::new(1000.0, sample_rate);
        
        // Test state classification
        let metrics = CavitationMetrics {
            intensity: 0.7,
            subharmonic_level: 0.4,
            broadband_level: 0.6,
            ..Default::default()
        };
        
        let state = detector.classify_state(&metrics);
        assert_eq!(state, CavitationState::Inertial);
    }
}