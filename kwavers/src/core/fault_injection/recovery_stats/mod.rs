// Statistical aggregation for recovery validation
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
// ### Theorem: Confidence Interval Coverage
// The Wald confidence interval for proportion p:
//
// ```text
// CI = p̂ ± z_(α/2) × √[p̂(1-p̂)/n]
// For α = 0.05, z = 1.96
// Coverage probability → 1-α as n → ∞
// ```

mod latency;
mod reports;
mod stats;
#[cfg(test)]
mod tests;

pub use latency::RecoveryLatencyStats;
pub use reports::{ContentionReport, RecoveryReport, StabilityReport, ThreadContentionMetrics};
pub use stats::RecoveryStats;

use super::scenario::FaultScenario;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

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
