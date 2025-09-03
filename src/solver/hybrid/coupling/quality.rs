//! Quality monitoring for interface coupling

use ndarray::Array3;
use std::collections::VecDeque;

const MAX_HISTORY_SIZE: usize = 100;

/// Quality monitor for interface coupling
#[derive(Debug)]
pub struct QualityMonitor {
    /// Quality metrics history
    history: VecDeque<InterfaceQualityMetrics>,
    /// Quality thresholds
    thresholds: QualityThresholds,
}

/// Interface quality metrics
#[derive(Debug, Clone)]
pub struct InterfaceQualityMetrics {
    /// Interpolation error
    pub interpolation_error: f64,
    /// Conservation error
    pub conservation_error: f64,
    /// Reflection coefficient
    pub reflection_coefficient: f64,
    /// Transmission coefficient
    pub transmission_coefficient: f64,
    /// Phase error
    pub phase_error: f64,
    /// Amplitude error
    pub amplitude_error: f64,
    /// Time stamp
    pub time: f64,
}

/// Quality thresholds
#[derive(Debug, Clone)]
struct QualityThresholds {
    max_interpolation_error: f64,
    max_conservation_error: f64,
    max_reflection: f64,
    min_transmission: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            max_interpolation_error: crate::physics::constants::numerical::INTERPOLATION_ERROR_THRESHOLD,
            max_conservation_error: crate::physics::constants::numerical::CONSERVATION_ERROR_THRESHOLD,
            max_reflection: 0.01,   // 1% reflection
            min_transmission: 0.99, // 99% transmission
        }
    }
}

impl Default for QualityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityMonitor {
    /// Create a new quality monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            thresholds: QualityThresholds::default(),
        }
    }

    /// Update quality metrics
    pub fn update(&mut self, interpolated: &Array3<f64>, target: &Array3<f64>, time: f64) {
        let metrics = self.calculate_metrics(interpolated, target, time);

        // Add to history
        if self.history.len() >= MAX_HISTORY_SIZE {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
    }

    /// Calculate quality metrics
    fn calculate_metrics(
        &self,
        interpolated: &Array3<f64>,
        target: &Array3<f64>,
        time: f64,
    ) -> InterfaceQualityMetrics {
        // Calculate interpolation error
        let interpolation_error = self.calculate_interpolation_error(interpolated, target);

        // Calculate conservation error
        let conservation_error = self.calculate_conservation_error(interpolated, target);

        // Calculate reflection and transmission coefficients
        let (reflection, transmission) = self.calculate_coefficients(interpolated, target);

        // Calculate phase and amplitude errors
        let phase_error = self.calculate_phase_error(interpolated, target);
        let amplitude_error = self.calculate_amplitude_error(interpolated, target);

        InterfaceQualityMetrics {
            interpolation_error,
            conservation_error,
            reflection_coefficient: reflection,
            transmission_coefficient: transmission,
            phase_error,
            amplitude_error,
            time,
        }
    }

    fn calculate_interpolation_error(
        &self,
        interpolated: &Array3<f64>,
        target: &Array3<f64>,
    ) -> f64 {
        let diff = interpolated - target;
        (diff.iter().map(|x| x * x).sum::<f64>() / diff.len() as f64).sqrt()
    }

    fn calculate_conservation_error(
        &self,
        interpolated: &Array3<f64>,
        target: &Array3<f64>,
    ) -> f64 {
        let source_sum: f64 = interpolated.iter().sum();
        let target_sum: f64 = target.iter().sum();
        (source_sum - target_sum).abs()
    }

    fn calculate_coefficients(
        &self,
        interpolated: &Array3<f64>,
        target: &Array3<f64>,
    ) -> (f64, f64) {
        let source_energy: f64 = interpolated.iter().map(|x| x * x).sum();
        let target_energy: f64 = target.iter().map(|x| x * x).sum();

        if source_energy > crate::physics::constants::numerical::ENERGY_THRESHOLD {
            let transmission = target_energy / source_energy;
            let reflection = 1.0 - transmission;
            (reflection.max(0.0), transmission.min(1.0))
        } else {
            (0.0, 1.0)
        }
    }

    fn calculate_phase_error(&self, interpolated: &Array3<f64>, target: &Array3<f64>) -> f64 {
        // Phase error calculation using cross-correlation peak shift
        // This measures the phase difference between the interpolated and target fields

        // Find the peak locations in both fields
        let interp_peak_idx = interpolated
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(idx, _)| idx);

        let target_peak_idx = target
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(idx, _)| idx);

        // Calculate phase shift as normalized position difference
        let total_size = interpolated.len();
        if total_size > 0 {
            (interp_peak_idx as f64 - target_peak_idx as f64).abs() / total_size as f64
        } else {
            0.0
        }
    }

    fn calculate_amplitude_error(&self, interpolated: &Array3<f64>, target: &Array3<f64>) -> f64 {
        let source_max = interpolated.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let target_max = target.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

        if source_max > crate::physics::constants::numerical::AMPLITUDE_THRESHOLD {
            (target_max - source_max).abs() / source_max
        } else {
            0.0
        }
    }

    /// Get current metrics
    #[must_use]
    pub fn get_metrics(&self) -> InterfaceQualityMetrics {
        self.history
            .back()
            .cloned()
            .unwrap_or(InterfaceQualityMetrics {
                interpolation_error: 0.0,
                conservation_error: 0.0,
                reflection_coefficient: 0.0,
                transmission_coefficient: 1.0,
                phase_error: 0.0,
                amplitude_error: 0.0,
                time: 0.0,
            })
    }

    /// Check if quality is acceptable
    #[must_use]
    pub fn is_quality_acceptable(&self) -> bool {
        if let Some(metrics) = self.history.back() {
            metrics.interpolation_error <= self.thresholds.max_interpolation_error
                && metrics.conservation_error <= self.thresholds.max_conservation_error
                && metrics.reflection_coefficient <= self.thresholds.max_reflection
                && metrics.transmission_coefficient >= self.thresholds.min_transmission
        } else {
            true
        }
    }
}

/// Interface quality summary
#[derive(Debug, Clone)]
pub struct InterfaceQualitySummary {
    /// Average metrics over time window
    pub average_metrics: InterfaceQualityMetrics,
    /// Maximum errors observed
    pub max_errors: InterfaceQualityMetrics,
    /// Quality assessment
    pub is_acceptable: bool,
}
