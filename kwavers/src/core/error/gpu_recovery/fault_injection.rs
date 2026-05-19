use crate::core::error::context::ErrorLocation;
use crate::core::error::KwaversError;
use crate::core::error::gpu::GpuError;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::context::RecoveryContext;
use super::manager::ErrorGpuRecoveryManager;
use super::telemetry::{
    DEVICE_LOST_LATENCY_BUDGET_MS, DEVICE_LOST_TARGET_RATE, OOM_LATENCY_BUDGET_MS, OOM_TARGET_RATE,
    TIMEOUT_LATENCY_BUDGET_MS, TIMEOUT_TARGET_RATE, VALIDATION_LATENCY_BUDGET_MS,
};

/// Fault scenario types for injection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuFaultScenario {
    DeviceLost,
    OutOfMemory,
    Timeout,
    Validation,
}

impl GpuFaultScenario {
    pub fn target_rate(&self) -> f64 {
        match self {
            GpuFaultScenario::DeviceLost => DEVICE_LOST_TARGET_RATE,
            GpuFaultScenario::OutOfMemory => OOM_TARGET_RATE,
            GpuFaultScenario::Timeout => TIMEOUT_TARGET_RATE,
            GpuFaultScenario::Validation => 0.95,
        }
    }

    pub fn latency_budget_ms(&self) -> u64 {
        match self {
            GpuFaultScenario::DeviceLost => DEVICE_LOST_LATENCY_BUDGET_MS,
            GpuFaultScenario::OutOfMemory => OOM_LATENCY_BUDGET_MS,
            GpuFaultScenario::Timeout => TIMEOUT_LATENCY_BUDGET_MS,
            GpuFaultScenario::Validation => VALIDATION_LATENCY_BUDGET_MS,
        }
    }

    pub fn to_error(&self) -> GpuError {
        match self {
            GpuFaultScenario::DeviceLost => GpuError::DeviceLost {
                operation: "fault_injection".to_string(),
            },
            GpuFaultScenario::OutOfMemory => GpuError::OutOfMemory {
                requested: 1024 * 1024 * 1024,
                available: 512 * 1024 * 1024,
                current: 100 * 1024 * 1024 * 10,
            },
            GpuFaultScenario::Timeout => GpuError::Timeout {
                operation: "fault_injection".to_string(),
                duration_ms: 5000,
            },
            GpuFaultScenario::Validation => GpuError::Validation {
                message: "fault injection validation error".to_string(),
            },
        }
    }
}

/// Fault injection configuration
#[derive(Debug, Clone)]
pub struct FaultInjectionConfig {
    pub trials_per_scenario: usize,
    pub validate_latency: bool,
    pub validate_causal_chain: bool,
}

impl Default for FaultInjectionConfig {
    fn default() -> Self {
        Self {
            trials_per_scenario: 1000,
            validate_latency: true,
            validate_causal_chain: true,
        }
    }
}

impl FaultInjectionConfig {
    pub fn with_trials(trials: usize) -> Self {
        Self {
            trials_per_scenario: trials,
            validate_latency: true,
            validate_causal_chain: true,
        }
    }
}

/// Result of a single fault injection trial
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub scenario: GpuFaultScenario,
    pub recovered: bool,
    pub latency: Duration,
    pub error: Option<std::sync::Arc<KwaversError>>,
    pub causal_preserved: bool,
}

/// Batch results for statistical validation
#[derive(Debug, Clone)]
pub struct FaultInjectionReport {
    pub results_by_scenario: HashMap<GpuFaultScenario, Vec<TrialResult>>,
    pub config: FaultInjectionConfig,
    pub start_time: Instant,
    pub end_time: Instant,
}

impl FaultInjectionReport {
    pub fn success_rate(&self, scenario: GpuFaultScenario) -> f64 {
        let results = self.results_by_scenario.get(&scenario);
        match results {
            None => 0.0,
            Some(r) => {
                if r.is_empty() {
                    0.0
                } else {
                    let success = r.iter().filter(|t| t.recovered).count();
                    success as f64 / r.len() as f64
                }
            }
        }
    }

    pub fn avg_latency(&self, scenario: GpuFaultScenario) -> Duration {
        let results = self.results_by_scenario.get(&scenario);
        match results {
            None => Duration::ZERO,
            Some(r) => {
                if r.is_empty() {
                    Duration::ZERO
                } else {
                    let total: u128 = r.iter().map(|t| t.latency.as_nanos()).sum();
                    Duration::from_nanos((total / r.len() as u128) as u64)
                }
            }
        }
    }

    pub fn meets_target(&self, scenario: GpuFaultScenario) -> bool {
        self.success_rate(scenario) >= scenario.target_rate()
    }

    pub fn meets_latency_budget(&self, scenario: GpuFaultScenario) -> bool {
        let avg = self.avg_latency(scenario);
        avg < Duration::from_millis(scenario.latency_budget_ms())
    }

    pub fn passed(&self) -> bool {
        self.results_by_scenario
            .keys()
            .all(|&s| self.meets_target(s) && self.meets_latency_budget(s))
    }

    pub fn summary(&self) -> String {
        let mut summary = String::from("Fault Injection Report\n");
        summary.push_str(&"=".repeat(50));
        summary.push('\n');

        for scenario in self.results_by_scenario.keys() {
            let rate = self.success_rate(*scenario);
            let latency = self.avg_latency(*scenario);
            let target = scenario.target_rate();
            let budget = scenario.latency_budget_ms();

            summary.push_str(&format!(
                "{:?}: rate={:.2}% (target {:.0}%), latency={:.2}ms (budget {}ms) {}\n",
                scenario,
                rate * 100.0,
                target * 100.0,
                latency.as_secs_f64() * 1000.0,
                budget,
                if self.meets_target(*scenario) {
                    "✓"
                } else {
                    "✗"
                }
            ));
        }

        summary.push_str(&format!(
            "\nOverall: {}\n",
            if self.passed() { "PASS" } else { "FAIL" }
        ));
        summary
    }

    pub fn execution_time(&self) -> Duration {
        self.end_time.duration_since(self.start_time)
    }
}

/// Fault injector for controlled testing
#[derive(Debug)]
pub struct GpuRecoveryFaultInjector {
    pub config: FaultInjectionConfig,
    recovery_manager: ErrorGpuRecoveryManager,
}

impl GpuRecoveryFaultInjector {
    pub fn new(config: FaultInjectionConfig) -> Self {
        Self {
            config,
            recovery_manager: ErrorGpuRecoveryManager::new(),
        }
    }

    pub fn with_manager(config: FaultInjectionConfig, manager: ErrorGpuRecoveryManager) -> Self {
        Self {
            config,
            recovery_manager: manager,
        }
    }

    pub fn inject_fault(&self, scenario: GpuFaultScenario) -> TrialResult {
        let error = scenario.to_error();
        let ctx = RecoveryContext::new(ErrorLocation::new("fault_injection.rs", 1, "inject"));

        let start = Instant::now();
        let result = self.recovery_manager.recover_gpu_error(&error, &ctx);
        let latency = start.elapsed();

        TrialResult {
            scenario,
            recovered: result.is_ok(),
            latency,
            error: result.err().map(std::sync::Arc::new),
            causal_preserved: true,
        }
    }

    pub fn run_suite(&self, scenarios: &[GpuFaultScenario]) -> FaultInjectionReport {
        let start_time = Instant::now();
        let mut results_by_scenario: HashMap<GpuFaultScenario, Vec<TrialResult>> = HashMap::new();

        for scenario in scenarios {
            let mut results = Vec::with_capacity(self.config.trials_per_scenario);

            for _ in 0..self.config.trials_per_scenario {
                let trial = self.inject_fault(*scenario);
                results.push(trial);
            }

            results_by_scenario.insert(*scenario, results);
        }

        FaultInjectionReport {
            results_by_scenario,
            config: self.config.clone(),
            start_time,
            end_time: Instant::now(),
        }
    }

    pub fn run_trials(&self, scenario: GpuFaultScenario, n: usize) -> Vec<TrialResult> {
        (0..n).map(|_| self.inject_fault(scenario)).collect()
    }

    pub fn calculate_stats(&self, results: &[TrialResult]) -> GpuRecoveryFaultStats {
        let n = results.len();
        let successes = results.iter().filter(|r| r.recovered).count();
        let rate = if n == 0 {
            0.0
        } else {
            successes as f64 / n as f64
        };

        let z = 1.96;
        let p = rate;
        let n_f = n as f64;
        let ci_low = (p + z * z / (2.0 * n_f)
            - z * ((p * (1.0 - p) + z * z / (4.0 * n_f)) / n_f).sqrt())
            / (1.0 + z * z / n_f);
        let ci_high =
            (p + z * z / (2.0 * n_f) + z * ((p * (1.0 - p) + z * z / (4.0 * n_f)) / n_f).sqrt())
                / (1.0 + z * z / n_f);

        GpuRecoveryFaultStats {
            trials: n,
            successes,
            rate,
            ci_low: ci_low.max(0.0),
            ci_high: ci_high.min(1.0),
        }
    }
}

/// Recovery statistics with confidence intervals
#[derive(Debug, Clone, Copy)]
pub struct GpuRecoveryFaultStats {
    pub trials: usize,
    pub successes: usize,
    pub rate: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

impl GpuRecoveryFaultStats {
    pub fn meets_target(&self, target: f64) -> bool {
        self.ci_low >= target
    }

    pub fn format(&self) -> String {
        format!(
            "{:.1}% [{:.1}%, {:.1}%] (n={})",
            self.rate * 100.0,
            self.ci_low * 100.0,
            self.ci_high * 100.0,
            self.trials
        )
    }
}
