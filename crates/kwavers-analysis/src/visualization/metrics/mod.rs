//! Visualization Metrics Module
//!
//! Provides performance metrics tracking for visualization.

use aequitas::systems::si::quantities::{Frequency, Time};
use std::collections::VecDeque;

const METRIC_HISTORY_SIZE: usize = 60;

/// Performance metrics for visualization
#[derive(Debug, Clone)]
pub struct VisualizationMetrics {
    /// Current frames per second
    pub frame_rate: Frequency,
    /// GPU memory usage in bytes
    pub gpu_memory_usage: usize,
    /// Average render duration.
    pub render_time: Time,
    /// Average data transfer duration.
    pub transfer_time: Time,
    /// Number of rendered primitives
    pub rendered_primitives: usize,
}

impl Default for VisualizationMetrics {
    fn default() -> Self {
        Self {
            frame_rate: Frequency::from_base(0.0),
            gpu_memory_usage: 0,
            render_time: Time::from_base(0.0),
            transfer_time: Time::from_base(0.0),
            rendered_primitives: 0,
        }
    }
}

/// Metrics tracker for performance monitoring
#[derive(Debug)]
pub struct MetricsTracker {
    /// History of render times
    render_times: VecDeque<Time>,
    /// History of transfer times
    transfer_times: VecDeque<Time>,
    /// History of frame-rate measurements.
    frame_rate_history: VecDeque<Frequency>,
    /// Current metrics
    current: VisualizationMetrics,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            render_times: VecDeque::with_capacity(METRIC_HISTORY_SIZE),
            transfer_times: VecDeque::with_capacity(METRIC_HISTORY_SIZE),
            frame_rate_history: VecDeque::with_capacity(METRIC_HISTORY_SIZE),
            current: VisualizationMetrics::default(),
        }
    }

    /// Update metrics with new measurements
    pub fn update(&mut self, render_time: Time, transfer_time: Time) {
        // Add to history
        if self.render_times.len() >= METRIC_HISTORY_SIZE {
            self.render_times.pop_front();
        }
        self.render_times.push_back(render_time);

        if self.transfer_times.len() >= METRIC_HISTORY_SIZE {
            self.transfer_times.pop_front();
        }
        self.transfer_times.push_back(transfer_time);

        // Calculate averages
        self.current.render_time = Time::from_base(
            self.render_times
                .iter()
                .map(|time| time.into_base())
                .sum::<f64>()
                / self.render_times.len() as f64,
        );
        self.current.transfer_time = Time::from_base(
            self.transfer_times
                .iter()
                .map(|time| time.into_base())
                .sum::<f64>()
                / self.transfer_times.len() as f64,
        );

        // Calculate frame rate from the measured frame duration.
        let total_frame_time = render_time.into_base() + transfer_time.into_base();
        if total_frame_time > 0.0 {
            let frame_rate = Frequency::from_base(1.0 / total_frame_time);
            if self.frame_rate_history.len() >= METRIC_HISTORY_SIZE {
                self.frame_rate_history.pop_front();
            }
            self.frame_rate_history.push_back(frame_rate);
            if !self.frame_rate_history.is_empty() {
                self.current.frame_rate = Frequency::from_base(
                    self.frame_rate_history
                        .iter()
                        .map(|rate| rate.into_base())
                        .sum::<f64>()
                        / self.frame_rate_history.len() as f64,
                );
            }
        }
    }

    /// Update GPU memory usage
    pub fn update_memory(&mut self, bytes: usize) {
        self.current.gpu_memory_usage = bytes;
    }

    /// Update primitive count
    pub fn update_primitives(&mut self, count: usize) {
        self.current.rendered_primitives = count;
    }

    /// Get current metrics
    pub fn current(&self) -> &VisualizationMetrics {
        &self.current
    }

    /// Check if performance meets the target frame rate.
    pub fn meets_target(&self, target_frame_rate: Frequency) -> bool {
        self.current.frame_rate.into_base() >= target_frame_rate.into_base() * 0.9
    }

    /// Get performance summary
    pub fn summary(&self) -> String {
        format!(
            "FPS: {:.1}, Render: {:.2}ms, Transfer: {:.2}ms, GPU Memory: {:.1}MB",
            self.current.frame_rate.into_base(),
            self.current.render_time.into_base() * 1_000.0,
            self.current.transfer_time.into_base() * 1_000.0,
            self.current.gpu_memory_usage as f64 / 1_048_576.0
        )
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.render_times.clear();
        self.transfer_times.clear();
        self.frame_rate_history.clear();
        self.current = VisualizationMetrics::default();
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}
