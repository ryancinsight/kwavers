//! Harmonic Generation and Tracking for KZK Equation
//!
//! This module implements tools for tracking harmonic content development
//! during nonlinear acoustic propagation, enabling monitoring of energy
//! transfer from fundamental to higher frequencies.
//!
//! ## Physical Mechanism
//!
//! **Harmonic Generation Process**:
//! When acoustic pressure propagates nonlinearly, the wave form steepens,
//! generating harmonic components (2f, 3f, ...). Energy cascades from
//! fundamental (1f) to higher harmonics as:
//!
//! ```
//! E_1f → E_2f + E_3f + ... (energy conservation)
//! E_total = E_1f + E_2f + E_3f + ...
//! ```
//!
//! **Physical Interpretation**:
//! - **Linear regime**: E_nf ≈ 0 for n > 1
//! - **Weakly nonlinear**: E_2f ∝ (pressure amplitude)²
//! - **Strongly nonlinear**: Energy distributed across many harmonics
//! - **Shock formation**: Infinite harmonics (discontinuity requires all frequencies)
//!
//! ## Harmonic Analysis Methods
//!
//! **1. Frequency Domain (FFT)**:
//! - Transform pressure to frequency domain
//! - Extract amplitude at harmonic frequencies (nf₀)
//! - Compute spectral content and THD
//!
//! **2. Time Domain (Waveform Analysis)**:
//! - Detect zero crossings and peaks
//! - Estimate waveform asymmetry (key harmonic indicator)
//! - Monitor peak-to-mean ratio
//!
//! **3. Predictive Models**:
//! - Weakly nonlinear: E_2f ≈ (B/2A) * (p₀/ρc₀)² * z
//! - Pierson's rule: Shock distance ∝ 1/(β*p₀)
//!
//! ## References
//!
//! - Aanonsen et al. (1984) "Distortion and harmonic generation in the nearfield"
//! - Tjøtta & Tjøtta (1980) "Nonlinear waves in fluids with viscosity and diffusivity"
//! - Zemp et al. (2004) "Modeling nonlinear ultrasound propagation"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Harmonic analysis configuration
#[derive(Debug, Clone, Copy)]
pub struct HarmonicConfig {
    /// Operating frequency (Hz)
    pub frequency: f64,

    /// Maximum harmonic to track (e.g., 5 tracks 1f-5f)
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

/// Prediction models for harmonic development
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
            frequency: 1e6, // 1 MHz
            max_harmonic: 5,
            sampling_rate: 100e6, // 100 MHz
            fft_size: 512,
            enable_spectral: true,
            enable_waveform: true,
            prediction_model: PredictionModel::Pierson,
            b_a: 3.5, // Water/tissue
        }
    }
}

/// Results from harmonic analysis
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

/// Harmonic tracker for nonlinear propagation
#[derive(Debug)]
pub struct HarmonicTracker {
    config: HarmonicConfig,
    history: Vec<HarmonicAnalysis>,
}

impl HarmonicTracker {
    /// Create new harmonic tracker
    pub fn new(config: HarmonicConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Analyze harmonics from pressure time series
    pub fn analyze_harmonics(&self, pressure: &Array1<f64>) -> KwaversResult<HarmonicAnalysis> {
        if pressure.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Pressure array is empty".to_string(),
            ));
        }

        let mut analysis = HarmonicAnalysis::default();

        // 1. Time-domain waveform analysis
        if self.config.enable_waveform {
            analysis = self.analyze_waveform(pressure, analysis)?;
        }

        // 2. Frequency-domain spectral analysis
        if self.config.enable_spectral {
            analysis = self.analyze_spectrum(pressure, analysis)?;
        }

        // 3. Predict shock distance
        if analysis.rms_fundamental > 0.0 {
            analysis.predicted_shock_distance =
                self.predict_shock_distance(analysis.rms_fundamental);
        }

        Ok(analysis)
    }

    /// Analyze waveform characteristics
    fn analyze_waveform(
        &self,
        pressure: &Array1<f64>,
        mut analysis: HarmonicAnalysis,
    ) -> KwaversResult<HarmonicAnalysis> {
        let n = pressure.len() as f64;

        // Compute RMS and peak values
        let mean = pressure.iter().sum::<f64>() / n;
        let variance = pressure.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let rms = variance.sqrt();

        analysis.rms_total = rms;

        // Crest factor (peak-to-RMS ratio)
        let peak = pressure.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if rms > 1e-10 {
            analysis.crest_factor = peak / rms;
        }

        // Skewness (asymmetry): E[(x-μ)³]/σ³
        if rms > 1e-10 {
            let m3 = pressure.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
            analysis.skewness = m3 / rms.powi(3);
        }

        // Kurtosis (peakedness): E[(x-μ)⁴]/σ⁴ - 3
        if rms > 1e-10 {
            let m4 = pressure.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
            analysis.kurtosis = m4 / rms.powi(4) - 3.0;
        }

        Ok(analysis)
    }

    /// Analyze harmonic content via spectrum
    fn analyze_spectrum(
        &self,
        pressure: &Array1<f64>,
        mut analysis: HarmonicAnalysis,
    ) -> KwaversResult<HarmonicAnalysis> {
        let _n = pressure.len() as f64;

        // Simple spectral analysis (in practice, use proper FFT)
        // Here we approximate by filtering pressure at harmonic frequencies

        let dt = 1.0 / self.config.sampling_rate;

        // Compute amplitudes at harmonic frequencies (simplified approach)
        let mut frequencies = Vec::new();
        let mut amplitudes = Vec::new();
        let mut power = Vec::new();

        for harmonic in 1..=self.config.max_harmonic {
            let freq = self.config.frequency * harmonic as f64;
            frequencies.push(freq);

            // Simple approach: filter at harmonic frequency
            let amplitude = self.extract_harmonic_amplitude(pressure, freq, dt)?;
            amplitudes.push(amplitude);
            power.push(amplitude * amplitude);
        }

        // Compute THD from harmonic content
        if !amplitudes.is_empty() && amplitudes[0] > 1e-10 {
            let fundamental_power = power[0];
            let harmonic_power: f64 = power.iter().skip(1).sum();

            analysis.rms_fundamental = amplitudes[0];
            analysis.thd = (harmonic_power / fundamental_power).sqrt() * 100.0;

            // Energy ratio: (2f+3f+...)/(1f)
            if fundamental_power > 0.0 {
                analysis.energy_ratio = harmonic_power / fundamental_power;
            }
        }

        analysis.frequencies = frequencies;
        analysis.amplitudes = amplitudes;
        analysis.power = power;

        Ok(analysis)
    }

    /// Extract harmonic amplitude at specific frequency
    fn extract_harmonic_amplitude(
        &self,
        pressure: &Array1<f64>,
        frequency: f64,
        dt: f64,
    ) -> KwaversResult<f64> {
        if pressure.is_empty() || frequency <= 0.0 {
            return Ok(0.0);
        }

        // Use Goertzel algorithm for single-frequency detection
        // Simplified: correlate with sine and cosine at target frequency

        let mut cos_acc = 0.0;
        let mut sin_acc = 0.0;

        for (i, &p) in pressure.iter().enumerate() {
            let phase = 2.0 * PI * frequency * (i as f64) * dt;
            cos_acc += p * phase.cos();
            sin_acc += p * phase.sin();
        }

        let amplitude = (cos_acc * cos_acc + sin_acc * sin_acc).sqrt() / pressure.len() as f64;
        Ok(amplitude)
    }

    /// Predict shock distance using Pierson's rule
    fn predict_shock_distance(&self, pressure_amplitude: f64) -> Option<f64> {
        if pressure_amplitude <= 0.0 {
            return None;
        }

        // Pierson's rule: z_shock ≈ ρ₀ c₀² / (β p₀)
        // where β = B/A * (1/2) is the nonlinearity parameter
        // For water/tissue: B/A ≈ 3.5, ρ₀ ≈ 1000 kg/m³, c₀ ≈ 1540 m/s

        let rho0 = 1000.0; // kg/m³
        let c0 = 1540.0; // m/s
        let beta = self.config.b_a / 2.0;

        let z_shock = rho0 * c0 * c0 / (beta * pressure_amplitude);

        Some(z_shock)
    }

    /// Record analysis result
    pub fn record_analysis(&mut self, analysis: HarmonicAnalysis) {
        self.history.push(analysis);
    }

    /// Get analysis history
    pub fn history(&self) -> &[HarmonicAnalysis] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get configuration
    pub fn config(&self) -> HarmonicConfig {
        self.config
    }

    /// Analyze spatial development of harmonics (2D field)
    pub fn analyze_harmonic_field(
        &self,
        pressure: &Array2<f64>,
    ) -> KwaversResult<Vec<HarmonicAnalysis>> {
        let (_nx, nz) = pressure.dim();
        let mut analyses = Vec::new();

        // Analyze harmonic content along propagation axis
        for z in 0..nz {
            let line = pressure.column(z).to_owned();
            if let Ok(analysis) = self.analyze_harmonics(&line) {
                analyses.push(analysis);
            }
        }

        Ok(analyses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_tracker_creation() {
        let config = HarmonicConfig::default();
        let _tracker = HarmonicTracker::new(config);
    }

    #[test]
    fn test_pure_sinusoid_analysis() {
        let config = HarmonicConfig::default();
        let tracker = HarmonicTracker::new(config);

        // Create pure sine wave at fundamental frequency
        let n = 1000;
        let dt = 1.0 / config.sampling_rate;
        let mut pressure = Array1::zeros(n);

        for i in 0..n {
            let phase = 2.0 * PI * config.frequency * (i as f64) * dt;
            pressure[i] = 100.0 * phase.sin();
        }

        let analysis = tracker.analyze_harmonics(&pressure).unwrap();
        assert!(analysis.rms_total > 0.0);
        assert!(analysis.crest_factor > 1.0);
        assert!(analysis.thd < 10.0); // Low THD for pure sine
    }

    #[test]
    fn test_harmonic_content_distorted_wave() {
        let config = HarmonicConfig::default();
        let tracker = HarmonicTracker::new(config);

        // Create distorted wave with harmonics
        let n = 1000;
        let dt = 1.0 / config.sampling_rate;
        let mut pressure = Array1::zeros(n);

        for i in 0..n {
            let phase = 2.0 * PI * config.frequency * (i as f64) * dt;
            // Fundamental + second harmonic
            pressure[i] = 100.0 * phase.sin() + 20.0 * (2.0 * phase).sin();
        }

        let analysis = tracker.analyze_harmonics(&pressure).unwrap();
        assert!(analysis.thd > 10.0); // Higher THD due to harmonics
        assert!(analysis.energy_ratio > 0.0);
    }

    #[test]
    fn test_shock_distance_prediction() {
        let config = HarmonicConfig::default();
        let tracker = HarmonicTracker::new(config);

        // High pressure should predict closer shock distance
        let distance_1mpa = tracker.predict_shock_distance(1e6);
        let distance_10mpa = tracker.predict_shock_distance(10e6);

        assert!(distance_1mpa.is_some());
        assert!(distance_10mpa.is_some());

        // Higher pressure = closer shock
        assert!(distance_10mpa.unwrap() < distance_1mpa.unwrap());
    }

    #[test]
    fn test_history_management() {
        let config = HarmonicConfig::default();
        let mut tracker = HarmonicTracker::new(config);

        let analysis = HarmonicAnalysis {
            thd: 5.0,
            ..Default::default()
        };

        tracker.record_analysis(analysis);
        assert_eq!(tracker.history().len(), 1);
        assert_eq!(tracker.history()[0].thd, 5.0);

        tracker.clear_history();
        assert_eq!(tracker.history().len(), 0);
    }

    #[test]
    fn test_config_validation() {
        let config = HarmonicConfig::default();
        assert!(config.frequency > 0.0);
        assert!(config.max_harmonic > 0);
        assert!(config.b_a > 0.0);
    }

    #[test]
    fn test_empty_pressure_handling() {
        let config = HarmonicConfig::default();
        let tracker = HarmonicTracker::new(config);
        let empty_pressure = Array1::zeros(0);

        let result = tracker.analyze_harmonics(&empty_pressure);
        assert!(result.is_err());
    }
}
