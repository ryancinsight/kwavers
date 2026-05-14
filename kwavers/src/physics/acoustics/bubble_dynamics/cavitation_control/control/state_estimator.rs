//! State estimation for control system

use super::super::detection::CavitationMetrics;
use std::collections::VecDeque;

/// State estimator for temporal smoothing
#[derive(Debug)]
pub struct StateEstimator {
    history: VecDeque<CavitationMetrics>,
    alpha: f64,
}

impl Default for StateEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl StateEstimator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(10),
            alpha: 0.3, // Smoothing factor
        }
    }
    /// Estimate.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn estimate(&mut self, metrics: &CavitationMetrics) -> CavitationMetrics {
        if self.history.is_empty() {
            self.history.push_back(metrics.clone());
            return metrics.clone();
        }

        let last = self.history.back().unwrap();

        // Exponential smoothing
        let smoothed = CavitationMetrics {
            intensity: self
                .alpha
                .mul_add(metrics.intensity, (1.0 - self.alpha) * last.intensity),
            subharmonic_level: self.alpha.mul_add(
                metrics.subharmonic_level,
                (1.0 - self.alpha) * last.subharmonic_level,
            ),
            ultraharmonic_level: self.alpha.mul_add(
                metrics.ultraharmonic_level,
                (1.0 - self.alpha) * last.ultraharmonic_level,
            ),
            broadband_level: self.alpha.mul_add(
                metrics.broadband_level,
                (1.0 - self.alpha) * last.broadband_level,
            ),
            harmonic_distortion: self.alpha.mul_add(
                metrics.harmonic_distortion,
                (1.0 - self.alpha) * last.harmonic_distortion,
            ),
            harmonic_content: self.alpha.mul_add(
                metrics.harmonic_content,
                (1.0 - self.alpha) * last.harmonic_content,
            ),
            cavitation_dose: metrics.cavitation_dose, // Don't smooth cumulative dose
            confidence: self
                .alpha
                .mul_add(metrics.confidence, (1.0 - self.alpha) * last.confidence),
            state: metrics.state, // Use current state
        };

        self.history.push_back(smoothed.clone());
        if self.history.len() > 10 {
            self.history.pop_front();
        }

        smoothed
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    #[must_use]
    pub fn get_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = self
            .history
            .iter()
            .rev()
            .take(5)
            .map(|m| m.intensity)
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear trend
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();

        n.mul_add(sum_xy, -(sum_x * sum_y)) / sum_x.mul_add(-sum_x, n * sum_x2)
    }
}
