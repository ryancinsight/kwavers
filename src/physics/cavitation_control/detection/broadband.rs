//! Broadband noise-based cavitation detection

use super::constants::*;
use super::traits::{CavitationDetector, DetectorParameters};
use super::types::{CavitationMetrics, CavitationState, DetectionMethod, HistoryBuffer};
use ndarray::ArrayView1;

/// Broadband detector for inertial cavitation
pub struct BroadbandDetector {
    sample_rate: f64,
    baseline_energy: Option<f64>,
    history: HistoryBuffer<f64>,
    sensitivity: f64,
}

impl BroadbandDetector {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            baseline_energy: None,
            history: HistoryBuffer::new(20),
            sensitivity: 1.0,
        }
    }

    /// Calculate signal energy
    fn calculate_energy(&self, signal: &ArrayView1<f64>) -> f64 {
        signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64
    }

    /// Detect broadband emissions
    fn detect_broadband_emissions(&mut self, signal: &ArrayView1<f64>) -> f64 {
        let current_energy = self.calculate_energy(signal);

        // Update baseline if not set
        if self.baseline_energy.is_none() {
            self.baseline_energy = Some(current_energy);
            return 0.0;
        }

        let baseline = self.baseline_energy.unwrap();

        // Calculate energy increase in dB
        if baseline > MIN_SPECTRAL_POWER {
            let db_increase = 10.0 * (current_energy / baseline).log10();

            // Update history
            self.history.push(db_increase);

            // Check if above threshold
            if db_increase > BROADBAND_THRESHOLD_DB * self.sensitivity {
                return (db_increase / (BROADBAND_THRESHOLD_DB * 3.0)).min(1.0);
            }
        }

        0.0
    }

    /// Update baseline for adaptive detection
    pub fn update_baseline(&mut self, signal: &ArrayView1<f64>) {
        self.baseline_energy = Some(self.calculate_energy(signal));
    }

    /// Apply temporal smoothing
    fn apply_temporal_smoothing(&self, current_value: f64) -> f64 {
        if self.history.is_empty() {
            return current_value;
        }

        let history_avg = self.history.iter().sum::<f64>() / self.history.len() as f64;
        TEMPORAL_SMOOTHING * current_value + (1.0 - TEMPORAL_SMOOTHING) * history_avg
    }
}

impl CavitationDetector for BroadbandDetector {
    fn detect(&mut self, signal: &ArrayView1<f64>) -> CavitationMetrics {
        let broadband_level = self.detect_broadband_emissions(signal);
        let smoothed_level = self.apply_temporal_smoothing(broadband_level);

        // Determine cavitation state
        let state = if smoothed_level > 0.5 {
            CavitationState::Inertial
        } else if smoothed_level > 0.2 {
            CavitationState::Transient
        } else {
            CavitationState::None
        };

        CavitationMetrics {
            state,
            subharmonic_level: 0.0,
            ultraharmonic_level: 0.0,
            broadband_level: smoothed_level,
            harmonic_distortion: 0.0,
            confidence: smoothed_level,
            // Legacy compatibility
            intensity: smoothed_level,
            harmonic_content: 0.0,
            cavitation_dose: 0.0,
        }
    }

    fn reset(&mut self) {
        self.baseline_energy = None;
        self.history = HistoryBuffer::new(20);
    }

    fn method(&self) -> DetectionMethod {
        DetectionMethod::Broadband
    }

    fn update_parameters(&mut self, params: DetectorParameters) {
        self.sample_rate = params.sample_rate;
        self.sensitivity = params.sensitivity;
    }
}
