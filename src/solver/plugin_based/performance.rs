//! Performance monitoring for plugin-based solver
//!
//! Tracks execution time, memory usage, and plugin performance metrics.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance metrics for a single plugin
#[derive(Debug, Clone, Default)]
pub struct PluginMetrics {
    /// Total execution time
    pub total_time: Duration,
    /// Number of executions
    pub execution_count: u64,
    /// Average execution time
    pub average_time: Duration,
    /// Peak execution time
    pub peak_time: Duration,
    /// Memory allocated (bytes)
    pub memory_allocated: usize,
}

/// Performance monitor for the solver
pub struct PerformanceMonitor {
    /// Per-plugin metrics
    plugin_metrics: HashMap<String, PluginMetrics>,
    /// Total solver execution time
    total_solver_time: Duration,
    /// Current iteration
    iteration: u64,
    /// Start time of current measurement
    current_start: Option<Instant>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            plugin_metrics: HashMap::new(),
            total_solver_time: Duration::ZERO,
            iteration: 0,
            current_start: None,
        }
    }

    /// Start timing for a plugin
    pub fn start_plugin(&mut self, plugin_name: &str) {
        self.current_start = Some(Instant::now());
    }

    /// End timing for a plugin
    pub fn end_plugin(&mut self, plugin_name: &str) {
        if let Some(start) = self.current_start.take() {
            let elapsed = start.elapsed();

            let metrics = self
                .plugin_metrics
                .entry(plugin_name.to_string())
                .or_default();

            metrics.total_time += elapsed;
            metrics.execution_count += 1;
            metrics.average_time = metrics.total_time / metrics.execution_count as u32;

            if elapsed > metrics.peak_time {
                metrics.peak_time = elapsed;
            }
        }
    }

    /// Record memory usage for a plugin
    pub fn record_memory(&mut self, plugin_name: &str, bytes: usize) {
        let metrics = self
            .plugin_metrics
            .entry(plugin_name.to_string())
            .or_default();
        metrics.memory_allocated = bytes;
    }

    /// Increment iteration counter
    pub fn next_iteration(&mut self) {
        self.iteration += 1;
    }

    /// Get current iteration
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Get metrics for a specific plugin
    pub fn plugin_metrics(&self, plugin_name: &str) -> Option<&PluginMetrics> {
        self.plugin_metrics.get(plugin_name)
    }

    /// Get all plugin metrics
    pub fn all_metrics(&self) -> &HashMap<String, PluginMetrics> {
        &self.plugin_metrics
    }

    /// Get total solver time
    pub fn total_time(&self) -> Duration {
        self.total_solver_time
    }

    /// Update total solver time
    pub fn update_total_time(&mut self, elapsed: Duration) {
        self.total_solver_time += elapsed;
    }

    /// Generate performance report
    pub fn report(&self) -> String {
        let mut report = format!("Performance Report (Iteration {})\n", self.iteration);
        report.push_str(&format!(
            "Total Solver Time: {:?}\n",
            self.total_solver_time
        ));
        report.push_str("\nPlugin Metrics:\n");

        for (name, metrics) in &self.plugin_metrics {
            report.push_str(&format!("  {}:\n", name));
            report.push_str(&format!("    Executions: {}\n", metrics.execution_count));
            report.push_str(&format!("    Total Time: {:?}\n", metrics.total_time));
            report.push_str(&format!("    Average Time: {:?}\n", metrics.average_time));
            report.push_str(&format!("    Peak Time: {:?}\n", metrics.peak_time));
            if metrics.memory_allocated > 0 {
                report.push_str(&format!("    Memory: {} bytes\n", metrics.memory_allocated));
            }
        }

        report
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.plugin_metrics.clear();
        self.total_solver_time = Duration::ZERO;
        self.iteration = 0;
        self.current_start = None;
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_performance_monitoring() {
        let mut monitor = PerformanceMonitor::new();

        // Simulate plugin execution
        monitor.start_plugin("acoustic");
        thread::sleep(Duration::from_millis(10));
        monitor.end_plugin("acoustic");

        monitor.start_plugin("thermal");
        thread::sleep(Duration::from_millis(5));
        monitor.end_plugin("thermal");

        // Check metrics
        let acoustic_metrics = monitor.plugin_metrics("acoustic").unwrap();
        assert_eq!(acoustic_metrics.execution_count, 1);
        assert!(acoustic_metrics.total_time >= Duration::from_millis(10));

        let thermal_metrics = monitor.plugin_metrics("thermal").unwrap();
        assert_eq!(thermal_metrics.execution_count, 1);
        assert!(thermal_metrics.total_time >= Duration::from_millis(5));

        // Test report generation
        let report = monitor.report();
        assert!(report.contains("acoustic"));
        assert!(report.contains("thermal"));
    }
}
