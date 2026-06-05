//! Metrics and monitoring for elastic wave simulations
//!
//! This module provides performance and physics metrics tracking
//! following SSOT principles.

use std::time::Duration;

/// Performance and physics metrics for elastic wave simulations
/// Follows SSOT principle - single source of simulation metrics
#[derive(Debug, Clone)]
pub struct ElasticWaveMetrics {
    pub total_steps: usize,
    pub fft_time: Duration,
    pub update_time: Duration,
    pub max_velocity: f64,
    pub max_stress: f64,
    pub energy: f64,
}

impl Default for ElasticWaveMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ElasticWaveMetrics {
    /// Create new metrics instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            fft_time: Duration::ZERO,
            update_time: Duration::ZERO,
            max_velocity: 0.0,
            max_stress: 0.0,
            energy: 0.0,
        }
    }

    /// Update maximum velocity if current value is larger
    pub fn update_max_velocity(&mut self, velocity: f64) {
        if velocity > self.max_velocity {
            self.max_velocity = velocity;
        }
    }

    /// Update maximum stress if current value is larger
    pub fn update_max_stress(&mut self, stress: f64) {
        if stress > self.max_stress {
            self.max_stress = stress;
        }
    }

    /// Add to FFT timing
    pub fn add_fft_time(&mut self, duration: Duration) {
        self.fft_time += duration;
    }

    /// Add to inverse FFT timing
    pub fn add_ifft_time(&mut self, duration: Duration) {
        self.fft_time += duration; // Count both FFT and IFFT in same metric
    }

    /// Add to update timing
    pub fn add_update_time(&mut self, duration: Duration) {
        self.update_time += duration;
    }

    /// Increment step counter
    pub fn increment_steps(&mut self) {
        self.total_steps += 1;
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.total_steps = 0;
        self.fft_time = Duration::ZERO;
        self.update_time = Duration::ZERO;
        self.max_velocity = 0.0;
        self.max_stress = 0.0;
        self.energy = 0.0;
    }

    /// Get average FFT time per step
    #[must_use]
    pub fn avg_fft_time(&self) -> Duration {
        if self.total_steps > 0 {
            self.fft_time / self.total_steps as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get average update time per step
    #[must_use]
    pub fn avg_update_time(&self) -> Duration {
        if self.total_steps > 0 {
            self.update_time / self.total_steps as u32
        } else {
            Duration::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::numerical::MPA_TO_PA;
    use std::time::Duration;

    #[test]
    fn new_is_fully_zeroed() {
        let m = ElasticWaveMetrics::new();
        assert_eq!(m.total_steps, 0);
        assert_eq!(m.fft_time, Duration::ZERO);
        assert_eq!(m.update_time, Duration::ZERO);
        assert_eq!(m.max_velocity, 0.0_f64);
        assert_eq!(m.max_stress, 0.0_f64);
        assert_eq!(m.energy, 0.0_f64);
    }

    /// `update_max_velocity` is a monotone-max: only updates when strictly greater.
    #[test]
    fn update_max_velocity_is_monotone() {
        let mut m = ElasticWaveMetrics::new();
        m.update_max_velocity(3.0);
        assert!((m.max_velocity - 3.0).abs() < 1e-15);
        m.update_max_velocity(1.0); // smaller — must not replace
        assert!((m.max_velocity - 3.0).abs() < 1e-15);
        m.update_max_velocity(5.0); // larger — must replace
        assert!((m.max_velocity - 5.0).abs() < 1e-15);
    }

    /// `update_max_stress` is a monotone-max: only updates when strictly greater.
    #[test]
    fn update_max_stress_is_monotone() {
        let mut m = ElasticWaveMetrics::new();
        m.update_max_stress(MPA_TO_PA);
        assert!((m.max_stress - MPA_TO_PA).abs() < 1.0);
        m.update_max_stress(0.5 * MPA_TO_PA);
        assert!((m.max_stress - MPA_TO_PA).abs() < 1.0);
        m.update_max_stress(2.0 * MPA_TO_PA);
        assert!((m.max_stress - 2.0 * MPA_TO_PA).abs() < 1.0);
    }

    #[test]
    fn add_fft_time_accumulates() {
        let mut m = ElasticWaveMetrics::new();
        m.add_fft_time(Duration::from_millis(10));
        m.add_fft_time(Duration::from_millis(20));
        assert_eq!(m.fft_time, Duration::from_millis(30));
    }

    /// `add_ifft_time` accumulates into the same `fft_time` bucket.
    #[test]
    fn add_ifft_time_accumulates_into_fft_metric() {
        let mut m = ElasticWaveMetrics::new();
        m.add_fft_time(Duration::from_millis(10));
        m.add_ifft_time(Duration::from_millis(5));
        assert_eq!(m.fft_time, Duration::from_millis(15));
    }

    #[test]
    fn add_update_time_accumulates_independently_of_fft() {
        let mut m = ElasticWaveMetrics::new();
        m.add_update_time(Duration::from_millis(7));
        m.add_update_time(Duration::from_millis(3));
        assert_eq!(m.update_time, Duration::from_millis(10));
        assert_eq!(m.fft_time, Duration::ZERO);
    }

    #[test]
    fn increment_steps_counts_correctly() {
        let mut m = ElasticWaveMetrics::new();
        for expected in 1..=5usize {
            m.increment_steps();
            assert_eq!(m.total_steps, expected);
        }
    }

    /// `avg_fft_time` returns `Duration::ZERO` when `total_steps == 0`.
    #[test]
    fn avg_fft_time_is_zero_with_no_steps() {
        let mut m = ElasticWaveMetrics::new();
        m.add_fft_time(Duration::from_millis(100));
        assert_eq!(m.avg_fft_time(), Duration::ZERO);
    }

    /// Analytically: 40 ms / 4 steps = 10 ms per step.
    #[test]
    fn avg_fft_time_divides_by_step_count() {
        let mut m = ElasticWaveMetrics::new();
        m.add_fft_time(Duration::from_millis(40));
        for _ in 0..4 {
            m.increment_steps();
        }
        assert_eq!(m.avg_fft_time(), Duration::from_millis(10));
    }

    /// Analytically: 60 ms / 3 steps = 20 ms per step.
    #[test]
    fn avg_update_time_divides_by_step_count() {
        let mut m = ElasticWaveMetrics::new();
        m.add_update_time(Duration::from_millis(60));
        for _ in 0..3 {
            m.increment_steps();
        }
        assert_eq!(m.avg_update_time(), Duration::from_millis(20));
    }

    #[test]
    fn reset_zeros_all_fields() {
        let mut m = ElasticWaveMetrics::new();
        m.increment_steps();
        m.update_max_velocity(9.0);
        m.update_max_stress(1e8);
        m.add_fft_time(Duration::from_millis(50));
        m.add_update_time(Duration::from_millis(25));
        m.reset();
        assert_eq!(m.total_steps, 0);
        assert_eq!(m.max_velocity, 0.0);
        assert_eq!(m.max_stress, 0.0);
        assert_eq!(m.fft_time, Duration::ZERO);
        assert_eq!(m.update_time, Duration::ZERO);
    }
}
