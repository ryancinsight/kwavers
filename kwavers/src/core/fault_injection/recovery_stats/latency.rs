use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Statistical distribution of recovery latencies
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecoveryLatencyStats {
    /// Number of samples
    pub n: usize,
    /// Sum of all latencies in nanoseconds
    pub sum_ns: u128,
    /// Sum of squared latencies (for variance)
    pub sum_sq_ns: u128,
    /// Minimum latency observed
    pub min_ns: u64,
    /// Maximum latency observed
    pub max_ns: u64,
}

impl RecoveryLatencyStats {
    /// Create empty stats
    pub fn new() -> Self {
        Self {
            n: 0,
            sum_ns: 0,
            sum_sq_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
        }
    }

    /// Add a latency sample using Welford's online algorithm
    pub fn add_sample(&mut self, duration: Duration) {
        let ns = duration.as_nanos();
        self.n += 1;
        self.sum_ns += ns;
        self.sum_sq_ns += ns * ns;
        let ns_u64 = duration.as_nanos() as u64;
        self.min_ns = self.min_ns.min(ns_u64);
        self.max_ns = self.max_ns.max(ns_u64);
    }

    /// Calculate mean latency
    pub fn mean(&self) -> Duration {
        if self.n == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos((self.sum_ns / self.n as u128) as u64)
        }
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            let mean = self.sum_ns as f64 / self.n as f64;
            let mean_sq = self.sum_sq_ns as f64 / self.n as f64;
            mean_sq - mean * mean
        }
    }

    /// Calculate standard deviation
    pub fn std_dev(&self) -> Duration {
        let variance = self.variance();
        Duration::from_nanos(variance.max(0.0_f32).sqrt() as u64)
    }

    /// Get p99 latency using normal approximation
    pub fn p99(&self) -> Duration {
        if self.n == 0 {
            return Duration::ZERO;
        }
        let mean_ns = self.mean().as_nanos() as f64;
        let std_ns = self.std_dev().as_nanos() as f64;
        // 99th percentile for normal distribution: mean + 2.33 * std
        Duration::from_nanos((mean_ns + 2.33 * std_ns) as u64)
    }

    /// Get p50 (median) approximation
    pub fn p50(&self) -> Duration {
        self.mean()
    }

    /// Get min latency
    pub fn min(&self) -> Duration {
        Duration::from_nanos(self.min_ns)
    }

    /// Get max latency
    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.max_ns)
    }
}
