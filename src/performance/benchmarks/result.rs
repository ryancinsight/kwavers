//! Benchmark result structures and reporting
//!
//! Provides data structures for storing and reporting benchmark results.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Individual benchmark result
#[derive(Debug, Clone))]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Grid size used
    pub grid_size: usize,
    /// Mean execution time
    pub mean_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Throughput (points/second)
    pub throughput: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(name: String, grid_size: usize, times: Vec<Duration>) -> Self {
        let mean_time = Self::calculate_mean(&times);
        let std_dev = Self::calculate_std_dev(&times, mean_time);
        let min_time = times.iter().min().copied().unwrap_or(Duration::ZERO);
        let max_time = times.iter().max().copied().unwrap_or(Duration::ZERO);

        let total_points = (grid_size * grid_size * grid_size) as f64;
        let throughput = total_points / mean_time.as_secs_f64();

        Self {
            name,
            grid_size,
            mean_time,
            std_dev,
            min_time,
            max_time,
            throughput,
            memory_usage: 0,
            metrics: HashMap::new(),
        }
    }

    fn calculate_mean(times: &[Duration]) -> Duration {
        if times.is_empty() {
            return Duration::ZERO;
        }

        let total: Duration = times.iter().sum();
        total / times.len() as u32
    }

    fn calculate_std_dev(times: &[Duration], mean: Duration) -> Duration {
        if times.len() <= 1 {
            return Duration::ZERO;
        }

        let mean_secs = mean.as_secs_f64();
        let variance = times
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean_secs;
                diff * diff
            })
            .sum::<f64>()
            / (times.len() - 1) as f64;

        Duration::from_secs_f64(variance.sqrt())
    }

    /// Add a custom metric
    pub fn add_metric(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }

    /// Get speedup relative to baseline
    pub fn speedup(&self, baseline: &BenchmarkResult) -> f64 {
        baseline.mean_time.as_secs_f64() / self.mean_time.as_secs_f64()
    }

    /// Get efficiency (speedup / cores)
    pub fn efficiency(&self, baseline: &BenchmarkResult, cores: usize) -> f64 {
        self.speedup(baseline) / cores as f64
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}x{} grid, {:.3}ms ± {:.3}ms, {:.2} Mpoints/s",
            self.name,
            self.grid_size,
            self.grid_size,
            self.mean_time.as_secs_f64() * 1000.0,
            self.std_dev.as_secs_f64() * 1000.0,
            self.throughput / 1e6
        )
    }
}

/// Benchmark report containing multiple results
#[derive(Debug, Clone))]
pub struct BenchmarkReport {
    /// All benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// System information
    pub system_info: HashMap<String, String>,
}

impl BenchmarkReport {
    /// Create a new benchmark report
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            timestamp: std::time::SystemTime::now(),
            system_info: Self::collect_system_info(),
        }
    }

    fn collect_system_info() -> HashMap<String, String> {
        let mut info = HashMap::new();

        // CPU info
        info.insert(
            "cpu_cores".to_string(),
            std::thread::available_parallelism()
                .map(|n| n.get().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
        );

        // Rust version (if available)
        info.insert("rust_version".to_string(), "stable".to_string());

        info
    }

    /// Add a result to the report
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Generate summary statistics
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("Benchmark Report - {:?}\n", self.timestamp));
        summary.push_str(&format!("Total benchmarks: {}\n", self.results.len()));

        if !self.system_info.is_empty() {
            summary.push_str("\nSystem Information:\n");
            for (key, value) in &self.system_info {
                summary.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        summary.push_str("\nResults:\n");
        for result in &self.results {
            summary.push_str(&format!("  {}\n", result));
        }

        summary
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("name,grid_size,mean_ms,std_dev_ms,throughput_mpoints\n");

        for result in &self.results {
            csv.push_str(&format!(
                "{},{},{:.3},{:.3},{:.2}\n",
                result.name,
                result.grid_size,
                result.mean_time.as_secs_f64() * 1000.0,
                result.std_dev.as_secs_f64() * 1000.0,
                result.throughput / 1e6
            ));
        }

        csv
    }

    /// Export to JSON format
    pub fn to_json(&self) -> String {
        // Simple JSON serialization without serde for now
        format!(
            "{{\"results\": [{}], \"timestamp\": \"{:?}\"}}",
            self.results
                .iter()
                .map(|r| format!(
                    "{{\"name\": \"{}\", \"grid_size\": {}, \"mean_ms\": {:.3}}}",
                    r.name,
                    r.grid_size,
                    r.mean_time.as_secs_f64() * 1000.0
                ))
                .collect::<Vec<_>>()
                .join(", "),
            self.timestamp
        )
    }
}

impl Default for BenchmarkReport {
    fn default() -> Self {
        Self::new()
    }
}
