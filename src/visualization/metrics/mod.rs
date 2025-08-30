//! Visualization Metrics Module
//!
//! Provides performance metrics tracking for visualization.

use std::collections::VecDeque;

const METRIC_HISTORY_SIZE: usize = 60;

/// Performance metrics for visualization
#[derive(Debug, Clone))]
pub struct VisualizationMetrics {
    /// Current frames per second
    pub fps: f64,
    /// GPU memory usage in bytes
    pub gpu_memory_usage: usize,
    /// Average render time in milliseconds
    pub render_time_ms: f32,
    /// Average data transfer time in milliseconds
    pub transfer_time_ms: f32,
    /// Number of rendered primitives
    pub rendered_primitives: usize,
}

impl Default for VisualizationMetrics {
    fn default() -> Self {
        Self {
            fps: 0.0,
            gpu_memory_usage: 0,
            render_time_ms: 0.0,
            transfer_time_ms: 0.0,
            rendered_primitives: 0,
        }
    }
}

/// Metrics tracker for performance monitoring
#[derive(Debug))]
pub struct MetricsTracker {
    /// History of render times
    render_times: VecDeque<f32>,
    /// History of transfer times
    transfer_times: VecDeque<f32>,
    /// History of FPS measurements
    fps_history: VecDeque<f64>,
    /// Current metrics
    current: VisualizationMetrics,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            render_times: VecDeque::with_capacity(METRIC_HISTORY_SIZE),
            transfer_times: VecDeque::with_capacity(METRIC_HISTORY_SIZE),
            fps_history: VecDeque::with_capacity(METRIC_HISTORY_SIZE),
            current: VisualizationMetrics::default(),
        }
    }

    /// Update metrics with new measurements
    pub fn update(&mut self, render_time: f32, transfer_time: f32) {
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
        self.current.render_time_ms =
            self.render_times.iter().sum::<f32>() / self.render_times.len() as f32;
        self.current.transfer_time_ms =
            self.transfer_times.iter().sum::<f32>() / self.transfer_times.len() as f32;

        // Calculate FPS
        let total_frame_time = render_time + transfer_time;
        if total_frame_time > 0.0 {
            let fps = 1000.0 / total_frame_time as f64;
            if self.fps_history.len() >= METRIC_HISTORY_SIZE {
                self.fps_history.pop_front();
            }
            self.fps_history.push_back(fps);
            self.current.fps = self.fps_history.iter().sum::<f64>() / self.fps_history.len() as f64;
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

    /// Check if performance meets target FPS
    pub fn meets_target(&self, target_fps: f64) -> bool {
        self.current.fps >= target_fps * 0.9 // Allow 10% tolerance
    }

    /// Get performance summary
    pub fn summary(&self) -> String {
        format!(
            "FPS: {:.1}, Render: {:.2}ms, Transfer: {:.2}ms, GPU Memory: {:.1}MB",
            self.current.fps,
            self.current.render_time_ms,
            self.current.transfer_time_ms,
            self.current.gpu_memory_usage as f64 / 1_048_576.0
        )
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.render_times.clear();
        self.transfer_times.clear();
        self.fps_history.clear();
        self.current = VisualizationMetrics::default();
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}
