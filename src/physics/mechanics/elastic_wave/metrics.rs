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
    pub fn avg_fft_time(&self) -> Duration {
        if self.total_steps > 0 {
            self.fft_time / self.total_steps as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get average update time per step
    pub fn avg_update_time(&self) -> Duration {
        if self.total_steps > 0 {
            self.update_time / self.total_steps as u32
        } else {
            Duration::ZERO
        }
    }
}
