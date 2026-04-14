// Statistical aggregation for recovery validation
//
// Provides comprehensive metrics and statistical analysis of recovery
// performance under fault conditions.
//
// ## Mathematical Foundations
//
// ### Theorem: Statistical Validity of Recovery Rate Estimation
// For n independent Bernoulli trials with success probability p,
// the sample proportion p̂ = k/n is an unbiased estimator of p:
//
// ```text
// E[p̂] = p
// Var(p̂) = p(1-p)/n
// ```
//
// By the Central Limit Theorem, for large n:
// p̂ ~ N(p, √(p(1-p)/n))
//
// ### Theorem: Confidence Interval Coverage
// The Wald confidence interval for proportion p:
//
// ```text
// CI = p̂ ± z_(α/2) × √[p̂(1-p̂)/n]
//
// For α = 0.05, z = 1.96
// Coverage probability → 1-α as n → ∞
// ```

use super::scenario::FaultScenario;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

/// Distribution of recovery outcomes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecoveryDistribution {
    /// Total number of recovery attempts
    pub total_attempts: u64,
    /// Number of successful recoveries
    pub successful_recoveries: u64,
    /// Number of failed recoveries
    pub failed_recoveries: u64,
    /// Failures by category
    pub failures_by_type: HashMap<String, u64>,
    /// Failures by strategy
    pub failures_by_strategy: HashMap<String, u64>,
}

/// Comprehensive telemetry integrity report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryIntegrity {
    /// Percentage of events successfully recorded [0.0, 100.0]
    pub event_capture_rate: f64,
    /// Causal chain preservation rate [0.0, 100.0]
    pub causal_chain_integrity: f64,
    /// Metric accuracy [0.0, 100.0]
    pub metric_accuracy: f64,
    /// Correlation with error timestamps
    pub timestamp_correlation: f64,
}

impl Default for TelemetryIntegrity {
    fn default() -> Self {
        Self {
            event_capture_rate: 100.0,
            causal_chain_integrity: 100.0,
            metric_accuracy: 100.0,
            timestamp_correlation: 1.0,
        }
    }
}

/// Recovery statistics with full telemetry
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecoveryStats {
    /// The fault scenario these stats apply to
    #[serde(skip)]
    pub scenario: Option<FaultScenario>,
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

/// Comprehensive recovery validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryReport {
    /// Overall statistics across all scenarios
    pub overall_stats: RecoveryStats,
    /// Per-scenario results
    pub scenario_results: Vec<super::ScenarioResult>,
    /// Total test duration
    pub total_duration: Duration,
    /// Whether overall threshold met
    pub overall_meets_threshold: bool,
    /// Minimum observed success rate
    pub min_success_rate: f64,
    /// Maximum observed success rate
    pub max_success_rate: f64,
}

impl RecoveryReport {
    /// Create a new empty recovery report
    pub fn new() -> Self {
        Self {
            overall_stats: RecoveryStats::default(),
            scenario_results: Vec::new(),
            total_duration: Duration::ZERO,
            overall_meets_threshold: false,
            min_success_rate: 1.0,
            max_success_rate: 0.0,
        }
    }

    /// Get success rates by scenario category
    pub fn success_rates_by_category(&self) -> HashMap<String, f64> {
        let mut rates: HashMap<String, Vec<f64>> = HashMap::new();

        for result in &self.scenario_results {
            let category = result.scenario.category().to_string();
            rates.entry(category).or_default().push(result.stats.success_rate());
        }

        rates.into_iter()
            .map(|(cat, vals)| {
                let avg = vals.iter().sum::<f64>() / vals.len() as f64;
                (cat, avg)
            })
            .collect()
    }
}

/// Stability report for long-duration runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    /// Total simulation steps executed
    pub total_steps: usize,
    /// Number of faults injected
    pub faults_injected: usize,
    /// Number of successful recoveries
    pub successful_recoveries: usize,
    /// Mean time between failures (MTBF) in seconds
    pub mtbf_seconds: f64,
    /// Mean time to recovery (MTTR) in microseconds
    pub mttr_microseconds: f64,
    /// Energy conservation violations
    pub energy_violations: usize,
    /// CFL stability violations
    pub cfl_violations: usize,
    /// Convergence failures
    pub convergence_failures: usize,
    /// Final simulation state integrity [0.0, 1.0]
    pub state_integrity: f64,
    /// Throughput stability (coefficient of variation)
    pub throughput_cv: f64,
    /// Memory usage trend (bytes/step)
    pub memory_trend: f64,
    /// Percentage of time system was stable
    pub stable_percentage: f64,
}

impl StabilityReport {
    /// Create new stability report
    pub fn new(total_steps: usize) -> Self {
        Self {
            total_steps,
            faults_injected: 0,
            successful_recoveries: 0,
            mtbf_seconds: f64::INFINITY,
            mttr_microseconds: 0.0,
            energy_violations: 0,
            cfl_violations: 0,
            convergence_failures: 0,
            state_integrity: 1.0,
            throughput_cv: 0.0,
            memory_trend: 0.0,
            stable_percentage: 100.0,
        }
    }

    /// Check if stability criteria met
    ///
    /// ## Stability Criteria
    /// - MTBF > 100s (failures infrequent relative to runtime)
    /// - MTTR < 100ms (recovery is fast)
    /// - Energy violations < 1 per 10k steps
    /// - Throughput CV < 5% (stable performance)
    /// - Memory trend stable (no leaks)
    pub fn is_stable(&self) -> bool {
        let criteria_met = self.mtbf_seconds > 100.0
            && self.mttr_microseconds < 100_000.0
            && (self.energy_violations as f64 / self.total_steps as f64) < 0.0001
            && self.throughput_cv < 0.05
            && self.memory_trend.abs() < 1.0
            && self.state_integrity > 0.99
            && self.stable_percentage >= 95.0;

        // Calculate stability score
        let mtbf_score = (self.mtbf_seconds / 100.0).min(1.0);
        let mttr_score = (100_000.0 / self.mttr_microseconds.max(1.0)).min(1.0);
        let energy_score = 1.0 - (self.energy_violations as f64 / self.total_steps as f64).min(1.0);
        let cv_score = 1.0 - (self.throughput_cv / 0.05).min(1.0);
        let memory_score = (1.0 - self.memory_trend.abs()).max(0.0_f32);
        let integrity_score = self.state_integrity;

        let overall_score = (mtbf_score + mttr_score + energy_score + cv_score + memory_score + integrity_score) / 6.0;

        criteria_met && overall_score >= 0.8
    }

    /// Get stability score [0.0, 1.0]
    pub fn stability_score(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }

        let recovery_rate = if self.faults_injected > 0 {
            self.successful_recoveries as f64 / self.faults_injected as f64
        } else {
            1.0
        };

        let mtbf_norm = (self.mtbf_seconds / 1000.0).min(1.0);
        let mttr_norm = (100_000.0 / self.mttr_microseconds.max(1000.0)).min(1.0);
        let integrity = self.state_integrity;

        (recovery_rate + mtbf_norm + mttr_norm + integrity) / 4.0
    }
}

/// Thread contention metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreadContentionMetrics {
    /// Thread ID or index
    pub thread_id: usize,
    /// Number of threads
    pub thread_count: usize,
    /// Contented lock acquisitions
    pub contented_locks: u64,
    /// Time spent waiting for locks (ms)
    pub lock_wait_time_ms: u64,
    /// Deadlock detections
    pub deadlocks: u64,
    /// Race conditions observed
    pub race_conditions: u64,
    /// Successful operations
    pub successful_ops: u64,
    /// Failed operations
    pub failed_ops: u64,
    /// Thread-local allocation efficiency [0.0, 1.0]
    pub local_allocation_efficiency: f64,
    /// Average operation latency (microseconds)
    pub avg_latency_us: f64,
}

/// Contention stress test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionReport {
    /// Thread metrics
    pub thread_metrics: Vec<ThreadContentionMetrics>,
    /// Aggregate throughput (ops/sec)
    pub aggregate_throughput: f64,
    /// Fairness index (Jain's fairness)
    pub fairness_index: f64,
    /// Scalability efficiency
    pub scalability: f64,
    /// Resource exhaustion events
    pub resource_exhaustions: usize,
    /// Recovery success rate under contention
    pub recovery_under_contention: f64,
    /// Deadlock count
    pub deadlock_count: usize,
    /// Race condition count
    pub race_count: usize,
    /// Test duration
    pub duration: Duration,
}

impl ContentionReport {
    /// Create new contention report
    pub fn new(thread_count: usize) -> Self {
        Self {
            thread_metrics: Vec::with_capacity(thread_count),
            aggregate_throughput: 0.0,
            fairness_index: 1.0,
            scalability: 1.0,
            resource_exhaustions: 0,
            recovery_under_contention: 1.0,
            deadlock_count: 0,
            race_count: 0,
            duration: Duration::ZERO,
        }
    }

    /// Calculate Jain's fairness index from throughputs
    ///
    /// Fairness F = (Σxᵢ)² / (n × Σxᵢ²)
    /// Where xᵢ is throughput of thread i, n is thread count
    /// F = 1 means perfect fairness, F → 1/n means worst fairness
    pub fn calculate_fairness(throughputs: &[f64]) -> f64 {
        if throughputs.is_empty() {
            return 1.0;
        }

        let sum: f64 = throughputs.iter().sum();
        let sum_sq: f64 = throughputs.iter().map(|x| x * x).sum();
        let n = throughputs.len() as f64;

        if sum_sq == 0.0 {
            return 1.0;
        }

        (sum * sum) / (n * sum_sq)
    }

    /// Check if contention is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.fairness_index >= 0.7
            && self.recovery_under_contention >= 0.85
            && self.deadlock_count == 0
            && self.race_count == 0
            && self.scalability >= 0.5
    }
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

    /// Calculate sample variance
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
        self.mean() // Approximation for symmetric distributions
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

impl RecoveryStats {
    /// Create stats for a specific scenario
    pub fn new(scenario: FaultScenario) -> Self {
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

        // Update causal preservation rate with exponential moving average
        let alpha = 0.1;
        let new_preserved = if causal_preserved { 1.0 } else { 0.0 };
        self.causal_preservation_rate = (1.0 - alpha) * self.causal_preservation_rate + alpha * new_preserved;
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

        // Merge failure maps
        for (k, v) in &other.distribution.failures_by_type {
            *self.distribution.failures_by_type.entry(k.clone()).or_insert(0) += v;
        }
        for (k, v) in &other.distribution.failures_by_strategy {
            *self.distribution.failures_by_strategy.entry(k.clone()).or_insert(0) += v;
        }

        // Merge latency stats
        self.latency.sum_ns += other.latency.sum_ns;
        self.latency.sum_sq_ns += other.latency.sum_sq_ns;
        self.latency.n += other.latency.n;
        self.latency.min_ns = self.latency.min_ns.min(other.latency.min_ns);
        self.latency.max_ns = self.latency.max_ns.max(other.latency.max_ns);

        // Weighted average of causal preservation rates
        let weighted_preserved = self.causal_preservation_rate * old_total as f64
            + other.causal_preservation_rate * other_total as f64;
        self.causal_preservation_rate = weighted_preserved / new_total as f64;
    }

    /// 95% confidence interval for success rate
    ///
    /// Uses Wilson score interval for better coverage with small samples
    pub fn confidence_interval(&self) -> (f64, f64) {
        let n = self.distribution.total_attempts as f64;
        if n == 0.0 {
            return (1.0, 1.0);
        }

        let successes = self.distribution.successful_recoveries as f64;
        let p = successes / n;

        // Wilson score interval
        let z = 1.96; // 95% CI
        let z_sq = z * z;
        let denominator = 1.0 + z_sq / n;

        let center = (p + z_sq / (2.0 * n)) / denominator;
        let margin = z * ((p * (1.0 - p) / n + z_sq / (4.0 * n * n)).sqrt()) / denominator;

        ((center - margin).max(0.0_f32), (center + margin).min(1.0))
    }

    /// Check if success rate meets threshold with statistical significance
    pub fn meets_threshold(&self, threshold: f64, confidence: f64) -> bool {
        let (lower, _upper) = self.confidence_interval();
        // One-sided test: lower bound must exceed threshold
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_stats_calculation() {
        let mut stats = RecoveryLatencyStats::new();
        stats.add_sample(Duration::from_millis(10));
        stats.add_sample(Duration::from_millis(20));
        stats.add_sample(Duration::from_millis(30));

        assert_eq!(stats.mean().as_millis(), 20);
        assert!(stats.std_dev().as_millis() > 0);
        assert_eq!(stats.n, 3);
        assert_eq!(stats.min().as_millis(), 10);
        assert_eq!(stats.max().as_millis(), 30);
    }

    #[test]
    fn recovery_stats_success_rate() {
        let mut stats = RecoveryStats::default();
        stats.record_trial(true, Duration::from_millis(10), true);
        stats.record_trial(true, Duration::from_millis(15), true);
        stats.record_trial(false, Duration::from_millis(20), true);

        let success_rate = stats.success_rate();
        assert!((success_rate - 2.0 / 3.0).abs() < 1e-10);

        let failure_rate = stats.failure_rate();
        assert!((failure_rate - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn stability_criteria() {
        // Stable system
        let stable = StabilityReport {
            total_steps: 1_000_000,
            faults_injected: 10,
            successful_recoveries: 10,
            mtbf_seconds: 1000.0,
            mttr_microseconds: 10_000.0,
            energy_violations: 0,
            cfl_violations: 0,
            convergence_failures: 0,
            state_integrity: 1.0,
            throughput_cv: 0.02,
            memory_trend: 0.0,
            stable_percentage: 100.0,
        };
        assert!(stable.is_stable());
        assert!(stable.stability_score() > 0.9);

        // Unstable system
        let unstable = StabilityReport {
            total_steps: 1_000_000,
            faults_injected: 1000,
            successful_recoveries: 500,
            mtbf_seconds: 10.0,
            mttr_microseconds: 500_000.0,
            energy_violations: 100,
            cfl_violations: 50,
            convergence_failures: 20,
            state_integrity: 0.5,
            throughput_cv: 0.2,
            memory_trend: 100.0,
            stable_percentage: 50.0,
        };
        assert!(!unstable.is_stable());
        assert!(unstable.stability_score() < 0.5);
    }

    #[test]
    fn fairness_index_calculation() {
        // Perfect fairness
        let fair = vec![100.0, 100.0, 100.0];
        assert!((ContentionReport::calculate_fairness(&fair) - 1.0).abs() < 1e-10);

        // Unfair distribution
        let unfair = vec![100.0, 50.0, 50.0];
        let fairness = ContentionReport::calculate_fairness(&unfair);
        assert!(fairness < 1.0);
        assert!(fairness > 0.0);

        // Minimum fairness: n threads, one has all throughput
        let min_fair = vec![100.0, 0.0, 0.0];
        let min_fairness = ContentionReport::calculate_fairness(&min_fair);
        assert!(min_fairness >= 0.33 && min_fairness <= 0.34); // Approx 1/3
    }

    #[test]
    fn confidence_interval_calculation() {
        let mut stats = RecoveryStats::default();

        // 950 successes out of 1000
        for _ in 0..950 {
            stats.record_trial(true, Duration::from_millis(10), true);
        }
        for _ in 0..50 {
            stats.record_trial(false, Duration::from_millis(10), true);
        }

        let (lower, upper) = stats.confidence_interval();
        assert!(lower < 0.95);
        assert!(upper > 0.95);
        assert!(lower > 0.90); // Should be tight at n=1000

        // Verify Wilson interval properties
        assert!(lower >= 0.0 && lower <= 1.0);
        assert!(upper >= 0.0 && upper <= 1.0);
        assert!(lower <= upper);
    }

    #[test]
    fn wilson_interval_properties() {
        // Edge cases
        let mut all_success = RecoveryStats::default();
        for _ in 0..100 {
            all_success.record_trial(true, Duration::from_millis(10), true);
        }
        let (lo, hi) = all_success.confidence_interval();
        assert!(lo > 0.95); // Should be very high

        let mut all_failure = RecoveryStats::default();
        for _ in 0..100 {
            all_failure.record_trial(false, Duration::from_millis(10), true);
        }
        let (lo, hi) = all_failure.confidence_interval();
        assert!(hi < 0.05); // Should be very low
        assert_eq!(lo, 0.0);
    }

    #[test]
    fn stats_merge() {
        let mut stats1 = RecoveryStats::new(FaultScenario::GpuOomSudden {
            allocation_size_bytes: 1024,
            timing: super::super::scenario::InjectionTiming::Immediate,
        });
        stats1.record_trial(true, Duration::from_millis(10), true);

        let mut stats2 = RecoveryStats::default();
        stats2.record_trial(false, Duration::from_millis(20), true);

        stats1.merge(&stats2);

        assert_eq!(stats1.distribution.total_attempts, 2);
        assert_eq!(stats1.distribution.successful_recoveries, 1);
        assert_eq!(stats1.distribution.failed_recoveries, 1);
    }
}
