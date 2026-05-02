use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use super::stats::RecoveryStats;

/// Comprehensive recovery validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryReport {
    /// Overall statistics across all scenarios
    pub overall_stats: RecoveryStats,
    /// Per-scenario results
    pub scenario_results: Vec<super::super::ScenarioResult>,
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
            rates
                .entry(category)
                .or_default()
                .push(result.stats.success_rate());
        }

        rates
            .into_iter()
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

    /// Check if stability criteria met.
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

        let mtbf_score = (self.mtbf_seconds / 100.0).min(1.0);
        let mttr_score = (100_000.0 / self.mttr_microseconds.max(1.0)).min(1.0);
        let energy_score =
            1.0 - (self.energy_violations as f64 / self.total_steps as f64).min(1.0);
        let cv_score = 1.0 - (self.throughput_cv / 0.05).min(1.0);
        let memory_score = (1.0 - self.memory_trend.abs()).max(0.0_f32);
        let integrity_score = self.state_integrity;

        let overall_score = (mtbf_score
            + mttr_score
            + energy_score
            + cv_score
            + memory_score
            + integrity_score)
            / 6.0;

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

    /// Calculate Jain's fairness index from throughputs.
    ///
    /// F = (Σxᵢ)² / (n × Σxᵢ²), where F = 1 means perfect fairness.
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
