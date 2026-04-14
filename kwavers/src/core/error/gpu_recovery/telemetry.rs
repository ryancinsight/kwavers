use std::sync::Mutex;
use std::time::Duration;
use tracing::debug;

/// Minimum acceptable success rate across all GPU recovery strategies.
/// Derived from system reliability requirements for continuous physics simulation.
/// P(Recovery | Failure) >= 0.90
pub const RECOVERY_SUCCESS_THRESHOLD: f64 = 0.90;

/// Target SLI for handling WGPU DeviceLost events.
/// Device loss must be nearly always recoverable (e.g., driver updates or hibernation).
/// P(Recovered | DeviceLost) >= 0.99
pub const DEVICE_LOST_TARGET_RATE: f64 = 0.99;

/// Target SLI for Out-Of-Memory (OOM) recovery.
/// Budget derived from probabilistic availability of host-mapped memory paging.
/// P(Recovered | OOM) >= 0.95
pub const OOM_TARGET_RATE: f64 = 0.95;

/// Target SLI for Compute Shader Timeout recovery (TDR).
/// Budget derived from acceptable frame-drop latency constraints.
/// P(Recovered | Timeout) >= 0.90
pub const TIMEOUT_TARGET_RATE: f64 = 0.90;

/// Strict latency budgets for GPU error recovery mechanisms.
/// Maximum bounds enforced before terminating the physics simulation pipeline.
pub const OOM_LATENCY_BUDGET_MS: u64 = 100;
pub const DEVICE_LOST_LATENCY_BUDGET_MS: u64 = 500;
pub const TIMEOUT_LATENCY_BUDGET_MS: u64 = 200;
pub const VALIDATION_LATENCY_BUDGET_MS: u64 = 50;

/// Backoff properties for temporal retry bounds
pub const MAX_TIMEOUT_RETRIES: u32 = 3;
pub const BASE_BACKOFF_MS: u64 = 10;
pub const MAX_BACKOFF_MS: u64 = 100;

/// Strategy success rates container
#[derive(Debug, Clone, Copy)]
pub struct GpuStrategyRates {
    pub device_lost: f64,
    pub oom: f64,
    pub timeout: f64,
}

impl GpuStrategyRates {
    pub fn meets_threshold(&self) -> bool {
        self.min_rate() >= RECOVERY_SUCCESS_THRESHOLD
            && self.device_lost >= DEVICE_LOST_TARGET_RATE
            && self.oom >= OOM_TARGET_RATE
            && self.timeout >= TIMEOUT_TARGET_RATE
    }

    pub fn min_rate(&self) -> f64 {
        self.device_lost.min(self.oom).min(self.timeout)
    }
}

/// Strategy latencies container
#[derive(Debug, Clone, Copy)]
pub struct GpuStrategyLatencies {
    pub device_lost: Duration,
    pub oom: Duration,
    pub timeout: Duration,
}

/// Global telemetry statistics (shared across all strategies)
#[derive(Debug, Clone, Default)]
pub struct GlobalTelemetry {
    pub total_attempts: usize,
    pub total_successes: usize,
    pub avg_latency_us: u64,
}

/// Complete telemetry snapshot for GPU recovery
#[derive(Debug, Clone)]
pub struct GpuRecoveryTelemetry {
    pub rates: GpuStrategyRates,
    pub latencies: GpuStrategyLatencies,
    pub global_stats: GlobalTelemetry,
}

// Global telemetry storage
pub(crate) static GLOBAL_TELEMETRY: std::sync::LazyLock<Mutex<GlobalTelemetry>> =
    std::sync::LazyLock::new(|| Mutex::new(GlobalTelemetry::default()));

/// Update global telemetry statistics
pub(crate) fn update_global_telemetry(strategy: &str, success: bool, latency_us: u64) {
    if let Ok(mut telemetry) = GLOBAL_TELEMETRY.lock() {
        telemetry.total_attempts += 1;
        if success {
            telemetry.total_successes += 1;
        }
        // Exponential moving average for latency
        let alpha = 0.1;
        telemetry.avg_latency_us =
            ((1.0 - alpha) * telemetry.avg_latency_us as f64 + alpha * latency_us as f64) as u64;

        debug!(
            strategy = strategy,
            success = success,
            latency_us = latency_us,
            "Updated global telemetry"
        );
    }
}

/// Get global telemetry snapshot
pub(crate) fn global_telemetry_snapshot() -> GlobalTelemetry {
    GLOBAL_TELEMETRY
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}
