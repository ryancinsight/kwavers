//! Timing profiling infrastructure
//!
//! Provides high-precision timing measurements for performance analysis.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Timing scope for RAII-based profiling
pub struct TimingScope {
    name: String,
    start: Instant,
    profiler: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

impl TimingScope {
    /// Create a new timing scope
    pub fn new(name: String, profiler: Arc<Mutex<HashMap<String, Vec<Duration>>>>) -> Self {
        Self {
            name,
            start: Instant::now(),
            profiler,
        }
    }
}

impl Drop for TimingScope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if let Ok(mut timings) = self.profiler.lock() {
            timings
                .entry(self.name.clone())
                .or_insert_with(Vec::new)
                .push(elapsed);
        }
    }
}

/// Summary statistics for timing measurements
#[derive(Debug, Clone)]
pub struct TimingSummary {
    /// Function or scope name
    pub name: String,
    /// Number of measurements
    pub count: usize,
    /// Total time
    pub total: Duration,
    /// Mean time
    pub mean: Duration,
    /// Minimum time
    pub min: Duration,
    /// Maximum time
    pub max: Duration,
    /// Standard deviation
    pub std_dev: Duration,
}

impl TimingSummary {
    /// Create a new timing summary from measurements
    pub fn from_measurements(name: String, measurements: &[Duration]) -> Self {
        if measurements.is_empty() {
            return Self {
                name,
                count: 0,
                total: Duration::ZERO,
                mean: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                std_dev: Duration::ZERO,
            };
        }

        let count = measurements.len();
        let total: Duration = measurements.iter().sum();
        let mean = total / count as u32;
        let min = *measurements.iter().min().unwrap();
        let max = *measurements.iter().max().unwrap();

        // Calculate standard deviation
        let mean_secs = mean.as_secs_f64();
        let variance = measurements
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean_secs;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        Self {
            name,
            count,
            total,
            mean,
            min,
            max,
            std_dev,
        }
    }

    /// Get relative percentage of total time
    pub fn percentage_of(&self, total: Duration) -> f64 {
        if total.as_nanos() == 0 {
            0.0
        } else {
            (self.total.as_nanos() as f64 / total.as_nanos() as f64) * 100.0
        }
    }
}

/// Timing profiler for collecting timing measurements
#[derive(Debug, Clone)]
pub struct TimingProfiler {
    timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

impl TimingProfiler {
    /// Create a new timing profiler
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start a timing scope
    pub fn scope(&self, name: &str) -> TimingScope {
        TimingScope::new(name.to_string(), self.timings.clone())
    }

    /// Record a timing measurement directly
    pub fn record(&self, name: &str, duration: Duration) {
        if let Ok(mut timings) = self.timings.lock() {
            timings
                .entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    /// Get all timing summaries
    pub fn summaries(&self) -> Vec<TimingSummary> {
        if let Ok(timings) = self.timings.lock() {
            timings
                .iter()
                .map(|(name, measurements)| {
                    TimingSummary::from_measurements(name.clone(), measurements)
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Clear all timing data
    pub fn clear(&self) {
        if let Ok(mut timings) = self.timings.lock() {
            timings.clear();
        }
    }
}

impl Default for TimingProfiler {
    fn default() -> Self {
        Self::new()
    }
}
