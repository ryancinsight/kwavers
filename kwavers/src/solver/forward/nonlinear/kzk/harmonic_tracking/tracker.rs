//! HarmonicTracker implementation.

use super::types::{HarmonicAnalysis, HarmonicConfig};
use crate::core::constants::fundamental::DENSITY_WATER_NOMINAL;
use crate::core::constants::numerical::TWO_PI;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};

/// Harmonic tracker for nonlinear propagation.
#[derive(Debug)]
pub struct HarmonicTracker {
    config: HarmonicConfig,
    history: Vec<HarmonicAnalysis>,
}

impl HarmonicTracker {
    /// Create new harmonic tracker.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(config: HarmonicConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Analyze harmonics from pressure time series.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn analyze_harmonics(&self, pressure: &Array1<f64>) -> KwaversResult<HarmonicAnalysis> {
        if pressure.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Pressure array is empty".to_owned(),
            ));
        }

        let mut analysis = HarmonicAnalysis::default();

        if self.config.enable_waveform {
            analysis = self.analyze_waveform(pressure, analysis)?;
        }

        if self.config.enable_spectral {
            analysis = self.analyze_spectrum(pressure, analysis)?;
        }

        if analysis.rms_fundamental > 0.0 {
            analysis.predicted_shock_distance =
                self.predict_shock_distance(analysis.rms_fundamental);
        }

        Ok(analysis)
    }

    /// Record analysis result.
    pub fn record_analysis(&mut self, analysis: HarmonicAnalysis) {
        self.history.push(analysis);
    }

    /// Get analysis history.
    #[must_use]
    pub fn history(&self) -> &[HarmonicAnalysis] {
        &self.history
    }

    /// Clear history.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get configuration.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn config(&self) -> HarmonicConfig {
        self.config
    }

    /// Analyze spatial development of harmonics (2D field).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn analyze_harmonic_field(
        &self,
        pressure: &Array2<f64>,
    ) -> KwaversResult<Vec<HarmonicAnalysis>> {
        let (_nx, nz) = pressure.dim();
        let mut analyses = Vec::new();

        for z in 0..nz {
            let line = pressure.column(z).to_owned();
            if let Ok(analysis) = self.analyze_harmonics(&line) {
                analyses.push(analysis);
            }
        }

        Ok(analyses)
    }

    // --- Private helpers ---

    fn analyze_waveform(
        &self,
        pressure: &Array1<f64>,
        mut analysis: HarmonicAnalysis,
    ) -> KwaversResult<HarmonicAnalysis> {
        let n = pressure.len() as f64;

        let mean = pressure.iter().sum::<f64>() / n;
        let variance = pressure.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let rms = variance.sqrt();

        analysis.rms_total = rms;

        let peak = pressure.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if rms > 1e-10 {
            analysis.crest_factor = peak / rms;
        }

        if rms > 1e-10 {
            let m3 = pressure.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
            analysis.skewness = m3 / rms.powi(3);
        }

        if rms > 1e-10 {
            let m4 = pressure.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
            analysis.kurtosis = m4 / rms.powi(4) - 3.0;
        }

        Ok(analysis)
    }

    fn analyze_spectrum(
        &self,
        pressure: &Array1<f64>,
        mut analysis: HarmonicAnalysis,
    ) -> KwaversResult<HarmonicAnalysis> {
        let _n = pressure.len() as f64;
        let dt = 1.0 / self.config.sampling_rate;

        let mut frequencies = Vec::new();
        let mut amplitudes = Vec::new();
        let mut power = Vec::new();

        for harmonic in 1..=self.config.max_harmonic {
            let freq = self.config.frequency * harmonic as f64;
            frequencies.push(freq);

            let amplitude = self.extract_harmonic_amplitude(pressure, freq, dt)?;
            amplitudes.push(amplitude);
            power.push(amplitude * amplitude);
        }

        if !amplitudes.is_empty() && amplitudes[0] > 1e-10 {
            let fundamental_power = power[0];
            let harmonic_power: f64 = power.iter().skip(1).sum();

            analysis.rms_fundamental = amplitudes[0];
            analysis.thd = (harmonic_power / fundamental_power).sqrt() * 100.0;

            if fundamental_power > 0.0 {
                analysis.energy_ratio = harmonic_power / fundamental_power;
            }
        }

        analysis.frequencies = frequencies;
        analysis.amplitudes = amplitudes;
        analysis.power = power;

        Ok(analysis)
    }

    fn extract_harmonic_amplitude(
        &self,
        pressure: &Array1<f64>,
        frequency: f64,
        dt: f64,
    ) -> KwaversResult<f64> {
        if pressure.is_empty() || frequency <= 0.0 {
            return Ok(0.0);
        }

        let mut cos_acc = 0.0;
        let mut sin_acc = 0.0;

        for (i, &p) in pressure.iter().enumerate() {
            let phase = TWO_PI * frequency * (i as f64) * dt;
            cos_acc += p * phase.cos();
            sin_acc += p * phase.sin();
        }

        let amplitude = cos_acc.hypot(sin_acc) / pressure.len() as f64;
        Ok(amplitude)
    }

    pub(super) fn predict_shock_distance(&self, pressure_amplitude: f64) -> Option<f64> {
        if pressure_amplitude <= 0.0 {
            return None;
        }

        // Fubini shock distance (Hamilton & Blackstock 1998, §4.3, eq. 4.3.5):
        //   z_shock = ρ₀ c₀³ / (β · ω₀ · p₀)
        // where β = 1 + B/(2A) and ω₀ = 2πf₀.
        let rho0: f64 = DENSITY_WATER_NOMINAL; // kg/m³
        let c0: f64 = crate::core::constants::fundamental::SOUND_SPEED_TISSUE; // m/s
        let omega0: f64 = TWO_PI * self.config.frequency;
        let beta: f64 = 1.0 + self.config.b_a / 2.0; // β = 1 + B/(2A)

        let z_shock = rho0 * f64::powi(c0, 3) / (beta * omega0 * pressure_amplitude);

        Some(z_shock)
    }
}
