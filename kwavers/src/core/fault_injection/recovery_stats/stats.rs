use serde::{Deserialize, Serialize};
use std::time::Duration;
use super::{FaultInjectionScenario, RecoveryDistribution, RecoveryLatencyStats, TelemetryIntegrity};

/// Recovery statistics with full telemetry
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecoveryStats {
    /// The fault scenario these stats apply to
    #[serde(skip)]
    pub scenario: Option<FaultInjectionScenario>,
    /// Distribution statistics
    pub distribution: RecoveryDistribution,
    /// Latency statistics
    pub latency: RecoveryLatencyStats,
    /// Telemetry integrity metrics
    pub telemetry: Option<TelemetryIntegrity>,
    /// Causal chain preservation rate [0.0, 1.0]
    pub causal_preservation_rate: f64,
    /// Sum of squared errors for online variance calculation
    sum_squared_errors: f64,
}

impl RecoveryStats {
    /// Create stats for a specific scenario
    pub fn new(scenario: FaultInjectionScenario) -> Self {
        Self {
            scenario: Some(scenario),
            distribution: RecoveryDistribution::default(),
            latency: RecoveryLatencyStats::new(),
            telemetry: None,
            causal_preservation_rate: 1.0,
            sum_squared_errors: 0.0,
        }
    }

    /// Record a single trial result
    pub fn record_trial(&mut self, succeeded: bool, latency: Duration, causal_preserved: bool) {
        self.distribution.total_attempts += 1;
        if succeeded {
            self.distribution.successful_recoveries += 1;
        } else {
            self.distribution.failed_recoveries += 1;
        }
        self.latency.add_sample(latency);

        let alpha = 0.1;
        let new_preserved = if causal_preserved { 1.0 } else { 0.0 };
        self.causal_preservation_rate =
            (1.0 - alpha) * self.causal_preservation_rate + alpha * new_preserved;
    }

    /// Record a failure by type
    pub fn record_failure_type(&mut self, failure_type: &str) {
        *self
            .distribution
            .failures_by_type
            .entry(failure_type.to_string())
            .or_insert(0) += 1;
    }

    /// Record a failure by strategy
    pub fn record_strategy_failure(&mut self, strategy: &str) {
        *self
            .distribution
            .failures_by_strategy
            .entry(strategy.to_string())
            .or_insert(0) += 1;
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.distribution.total_attempts == 0 {
            1.0
        } else {
            self.distribution.successful_recoveries as f64
                / self.distribution.total_attempts as f64
        }
    }

    /// Calculate failure rate
    pub fn failure_rate(&self) -> f64 {
        if self.distribution.total_attempts == 0 {
            0.0
        } else {
            self.distribution.failed_recoveries as f64
                / self.distribution.total_attempts as f64
        }
    }

    /// Merge another stats object into this one
    pub fn merge(&mut self, other: &Self) {
        let old_total = self.distribution.total_attempts;
        let other_total = other.distribution.total_attempts;
        let new_total = old_total + other_total;

        if new_total == 0 {
            return;
        }

        self.distribution.total_attempts += other.distribution.total_attempts;
        self.distribution.successful_recoveries += other.distribution.successful_recoveries;
        self.distribution.failed_recoveries += other.distribution.failed_recoveries;

        for (k, v) in &other.distribution.failures_by_type {
            *self.distribution.failures_by_type.entry(k.clone()).or_insert(0) += v;
        }
        for (k, v) in &other.distribution.failures_by_strategy {
            *self
                .distribution
                .failures_by_strategy
                .entry(k.clone())
                .or_insert(0) += v;
        }

        self.latency.sum_ns += other.latency.sum_ns;
        self.latency.sum_sq_ns += other.latency.sum_sq_ns;
        self.latency.n += other.latency.n;
        self.latency.min_ns = self.latency.min_ns.min(other.latency.min_ns);
        self.latency.max_ns = self.latency.max_ns.max(other.latency.max_ns);

        let weighted_preserved = self.causal_preservation_rate * old_total as f64
            + other.causal_preservation_rate * other_total as f64;
        self.causal_preservation_rate = weighted_preserved / new_total as f64;
    }

    /// 95% confidence interval for success rate (Wilson score interval).
    pub fn confidence_interval(&self) -> (f64, f64) {
        let n = self.distribution.total_attempts as f64;
        if n == 0.0 {
            return (1.0, 1.0);
        }

        let successes = self.distribution.successful_recoveries as f64;
        let p = successes / n;

        let z = 1.96;
        let z_sq = z * z;
        let denominator = 1.0 + z_sq / n;

        let center = (p + z_sq / (2.0 * n)) / denominator;
        let margin = z * ((p * (1.0 - p) / n + z_sq / (4.0 * n * n)).sqrt()) / denominator;

        ((center - margin).max(0.0_f32), (center + margin).min(1.0))
    }

    /// Check if success rate meets threshold with statistical significance
    pub fn meets_threshold(&self, threshold: f64, _confidence: f64) -> bool {
        let (lower, _upper) = self.confidence_interval();
        lower >= threshold
    }

    /// Get throughput estimate (recoveries per second)
    pub fn throughput(&self) -> f64 {
        let total_time_secs = self.latency.sum_ns as f64 / 1e9;
        if total_time_secs > 0.0 {
            self.distribution.total_attempts as f64 / total_time_secs
        } else {
            0.0
        }
    }
}
