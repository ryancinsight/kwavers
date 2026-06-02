//! Types for harmonic tracking: config, model, and analysis result.

use kwavers_core::constants::numerical::MHZ_TO_HZ;

/// Harmonic analysis configuration.
#[derive(Debug, Clone, Copy)]
pub struct HarmonicConfig {
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Maximum harmonic to track (e.g., 5 tracks 1f–5f)
    pub max_harmonic: usize,
    /// Sampling rate for waveform analysis (Hz)
    pub sampling_rate: f64,
    /// FFT buffer size (should be power of 2)
    pub fft_size: usize,
    /// Enable spectral analysis
    pub enable_spectral: bool,
    /// Enable time-domain waveform analysis
    pub enable_waveform: bool,
    /// Prediction model (Pierson or weakly nonlinear)
    pub prediction_model: PredictionModel,
    /// Medium nonlinearity parameter (B/A)
    pub b_a: f64,
}

/// Prediction models for harmonic development.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionModel {
    /// Pierson's rule for shock distance
    Pierson,
    /// Weakly nonlinear approximation
    WeaklyNonlinear,
    /// Burgers equation model
    Burgers,
}

impl Default for HarmonicConfig {
    fn default() -> Self {
        Self {
            frequency: MHZ_TO_HZ, // 1 MHz
            max_harmonic: 5,
            sampling_rate: 100.0 * MHZ_TO_HZ, // 100 MHz
            fft_size: 512,
            enable_spectral: true,
            enable_waveform: true,
            prediction_model: PredictionModel::Pierson,
            b_a: 3.5, // Water/tissue
        }
    }
}

/// Results from harmonic analysis.
#[derive(Debug, Clone)]
pub struct HarmonicAnalysis {
    /// Harmonic frequencies analyzed (Hz)
    pub frequencies: Vec<f64>,
    /// Harmonic amplitudes (Pa)
    pub amplitudes: Vec<f64>,
    /// Harmonic power (Pa²)
    pub power: Vec<f64>,
    /// Total Harmonic Distortion (%)
    pub thd: f64,
    /// Fundamental RMS (Pa)
    pub rms_fundamental: f64,
    /// Total RMS (Pa)
    pub rms_total: f64,
    /// Crest factor (peak/RMS)
    pub crest_factor: f64,
    /// Waveform skewness (asymmetry indicator)
    pub skewness: f64,
    /// Kurtosis (peakedness indicator)
    pub kurtosis: f64,
    /// Predicted shock distance (m)
    pub predicted_shock_distance: Option<f64>,
    /// Energy ratio (2f+3f+...)/1f
    pub energy_ratio: f64,
}

impl Default for HarmonicAnalysis {
    fn default() -> Self {
        Self {
            frequencies: Vec::new(),
            amplitudes: Vec::new(),
            power: Vec::new(),
            thd: 0.0,
            rms_fundamental: 0.0,
            rms_total: 0.0,
            crest_factor: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            predicted_shock_distance: None,
            energy_ratio: 0.0,
        }
    }
}
