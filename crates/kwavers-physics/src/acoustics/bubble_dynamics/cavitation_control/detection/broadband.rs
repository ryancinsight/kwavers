//! Broadband noise-based cavitation detection

use super::constants::{BROADBAND_THRESHOLD_DB, MIN_SPECTRAL_POWER, TEMPORAL_SMOOTHING};
use super::traits::{CavitationDetector, DetectorParameters};
use super::types::{CavitationDetectionState, CavitationMetrics, DetectionMethod, HistoryBuffer};
use ndarray::ArrayView1;

/// Broadband detector for inertial cavitation
pub struct BroadbandDetector {
    sample_rate: f64,
    baseline_energy: Option<f64>,
    history: HistoryBuffer<f64>,
    sensitivity: f64,
}

impl std::fmt::Debug for BroadbandDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BroadbandDetector")
            .field("sample_rate", &self.sample_rate)
            .field("baseline_energy", &self.baseline_energy)
            .field("history", &"<HistoryBuffer>")
            .field("sensitivity", &self.sensitivity)
            .finish()
    }
}

impl BroadbandDetector {
    #[must_use]
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            baseline_energy: None,
            history: HistoryBuffer::new(20),
            sensitivity: 1.0,
        }
    }

    /// Calculate signal energy
    fn calculate_energy(&self, signal: &ArrayView1<f64>) -> Option<f64> {
        if signal.is_empty() || !signal.iter().all(|value| value.is_finite()) {
            return None;
        }

        let energy = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
        energy.is_finite().then_some(energy)
    }

    /// Detect broadband emissions
    fn detect_broadband_emissions(&mut self, signal: &ArrayView1<f64>) -> f64 {
        let Some(current_energy) = self.calculate_energy(signal) else {
            return 0.0;
        };

        // Update baseline if not set
        if self.baseline_energy.is_none() {
            self.baseline_energy = Some(current_energy);
            return 0.0;
        }

        let Some(baseline) = self.baseline_energy.filter(|value| value.is_finite()) else {
            self.baseline_energy = Some(current_energy);
            return 0.0;
        };

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
        self.baseline_energy = self.calculate_energy(signal);
    }

    /// Apply temporal smoothing
    fn apply_temporal_smoothing(&self, current_value: f64) -> f64 {
        if self.history.is_empty() {
            return current_value;
        }

        let history_avg = self.history.iter().sum::<f64>() / self.history.len() as f64;
        TEMPORAL_SMOOTHING.mul_add(current_value, (1.0 - TEMPORAL_SMOOTHING) * history_avg)
    }
}

impl CavitationDetector for BroadbandDetector {
    fn detect(&mut self, signal: &ArrayView1<f64>) -> CavitationMetrics {
        let broadband_level = self.detect_broadband_emissions(signal);
        let smoothed_level = self.apply_temporal_smoothing(broadband_level);

        // Determine cavitation state
        let state = if smoothed_level > 0.5 {
            CavitationDetectionState::Inertial
        } else if smoothed_level > 0.2 {
            CavitationDetectionState::Transient
        } else {
            CavitationDetectionState::None
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

#[cfg(test)]
mod tests {
    use super::BroadbandDetector;
    use crate::acoustics::bubble_dynamics::cavitation_control::detection::traits::CavitationDetector;
    use crate::acoustics::bubble_dynamics::cavitation_control::detection::types::CavitationDetectionState;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use ndarray::arr1;

    #[test]
    fn broadband_detector_rejects_empty_and_nonfinite_signals() {
        let mut detector = BroadbandDetector::new(MHZ_TO_HZ);

        let empty = arr1(&[]);
        let empty_metrics = detector.detect(&empty.view());
        assert_eq!(empty_metrics.state, CavitationDetectionState::None);
        assert_eq!(empty_metrics.broadband_level, 0.0);
        assert_eq!(empty_metrics.confidence, 0.0);
        assert!(detector.baseline_energy.is_none());

        let nonfinite = arr1(&[0.0, f64::NAN, 1.0]);
        detector.update_baseline(&nonfinite.view());
        assert!(detector.baseline_energy.is_none());

        let nonfinite_metrics = detector.detect(&nonfinite.view());
        assert_eq!(nonfinite_metrics.state, CavitationDetectionState::None);
        assert_eq!(nonfinite_metrics.broadband_level, 0.0);
        assert_eq!(nonfinite_metrics.confidence, 0.0);
        assert!(detector.baseline_energy.is_none());
    }

    #[test]
    fn broadband_detector_recovers_after_invalid_signal() {
        let mut detector = BroadbandDetector::new(MHZ_TO_HZ);
        let invalid = arr1(&[f64::INFINITY]);
        assert_eq!(detector.detect(&invalid.view()).confidence, 0.0);
        assert!(detector.baseline_energy.is_none());

        let baseline = arr1(&[1.0, -1.0, 1.0, -1.0]);
        let first_valid = detector.detect(&baseline.view());
        assert_eq!(first_valid.confidence, 0.0);
        assert_eq!(detector.baseline_energy, Some(1.0));

        let elevated = arr1(&[10.0, -10.0, 10.0, -10.0]);
        let elevated_metrics = detector.detect(&elevated.view());
        assert!(elevated_metrics.broadband_level > 0.0);
        assert!(elevated_metrics.confidence > 0.0);
    }
}
