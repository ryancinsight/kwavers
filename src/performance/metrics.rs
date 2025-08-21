//! Performance metrics tracking for plugin execution

use std::collections::HashMap;
use std::time::Duration;

/// Performance metrics tracker
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Plugin execution times
    plugin_times: HashMap<String, Duration>,
    /// Total execution time
    total_time: Duration,
}

impl PerformanceMetrics {
    /// Create a new performance metrics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record plugin execution time
    pub fn record_plugin_execution(&mut self, plugin_id: &str, duration: Duration) {
        self.plugin_times.insert(plugin_id.to_string(), duration);
    }

    /// Record total execution time
    pub fn record_total_execution(&mut self, duration: Duration) {
        self.total_time = duration;
    }

    /// Get plugin execution time
    pub fn get_plugin_time(&self, plugin_id: &str) -> Option<Duration> {
        self.plugin_times.get(plugin_id).copied()
    }

    /// Get total execution time
    pub fn get_total_time(&self) -> Duration {
        self.total_time
    }

    /// Get all plugin times
    pub fn get_all_plugin_times(&self) -> &HashMap<String, Duration> {
        &self.plugin_times
    }
}
