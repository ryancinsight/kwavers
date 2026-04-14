pub(crate) const RECOVERY_SUCCESS_THRESHOLD: f64 = 0.90;

// ── GPU Recovery Statistics ──────────────────────────────────────────────────

/// Maximum number of retry attempts for timeout recovery
pub(crate) const MAX_TIMEOUT_RETRIES: u32 = 3;

/// Base delay for exponential backoff (milliseconds)
pub(crate) const BASE_BACKOFF_MS: u64 = 10;

/// Maximum backoff delay (milliseconds)
pub(crate) const MAX_BACKOFF_MS: u64 = 100;

/// GPU recovery statistics for telemetry
#[derive(Debug, Clone, Default)]
pub struct GpuRecoveryStats {
    /// Total attempts for device lost recovery
    pub device_lost_attempts: usize,
    /// Successful device lost recoveries
    pub device_lost_successes: usize,
    /// Total attempts for OOM recovery
    pub oom_attempts: usize,
    /// Successful OOM recoveries
    pub oom_successes: usize,
    /// Total attempts for timeout recovery
    pub timeout_attempts: usize,
    /// Successful timeout recoveries
    pub timeout_successes: usize,
    /// Average recovery latency (microseconds)
    pub avg_latency_us: u64,
}

impl GpuRecoveryStats {
    /// Calculate success rate for device lost recovery
    pub fn device_lost_rate(&self) -> f64 {
        if self.device_lost_attempts == 0 {
            1.0 // No attempts = assume success
        } else {
            self.device_lost_successes as f64 / self.device_lost_attempts as f64
        }
    }

    /// Calculate success rate for OOM recovery
    pub fn oom_rate(&self) -> f64 {
        if self.oom_attempts == 0 {
            1.0
        } else {
            self.oom_successes as f64 / self.oom_attempts as f64
        }
    }

    /// Calculate success rate for timeout recovery
    pub fn timeout_rate(&self) -> f64 {
        if self.timeout_attempts == 0 {
            1.0
        } else {
            self.timeout_successes as f64 / self.timeout_attempts as f64
        }
    }

    /// Overall success rate across all strategies
    pub fn overall_rate(&self) -> f64 {
        let total = self.device_lost_attempts + self.oom_attempts + self.timeout_attempts;
        let success = self.device_lost_successes + self.oom_successes + self.timeout_successes;

        if total == 0 {
            1.0
        } else {
            success as f64 / total as f64
        }
    }

    /// Check if all strategies meet threshold
    pub fn meets_threshold(&self) -> bool {
        self.device_lost_rate() >= RECOVERY_SUCCESS_THRESHOLD
            && self.oom_rate() >= RECOVERY_SUCCESS_THRESHOLD
            && self.timeout_rate() >= RECOVERY_SUCCESS_THRESHOLD
    }
}

/// Global GPU recovery statistics (shared across strategies)

/// Global GPU recovery statistics (shared across strategies)
pub static GLOBAL_STATS: std::sync::LazyLock<std::sync::Mutex<GpuRecoveryStats>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(GpuRecoveryStats::default()));

pub fn update_avg_latency_us(stats: &mut GpuRecoveryStats, latency_us: u64) {
    let total_attempts = stats.device_lost_attempts + stats.oom_attempts + stats.timeout_attempts;
    if total_attempts == 0 {
        stats.avg_latency_us = 0;
        return;
    }

    let weighted_total = stats
        .avg_latency_us
        .saturating_mul(total_attempts.saturating_sub(1) as u64);
    stats.avg_latency_us = weighted_total
        .saturating_add(latency_us)
        .checked_div(total_attempts as u64)
        .unwrap_or(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::recovery::RecoveryStrategy;
    use crate::core::error::{ErrorContext, KwaversError, SystemError};
    use crate::gpu::recovery::{DeviceLostRecovery, GpuCheckpoint, GpuRecoveryManager};
    use std::sync::{Arc, Mutex};

    #[test]
    fn global_stats_update() {
        // Test that global stats are updated
        let before = GpuRecoveryManager::global_stats();

        // Use a recovery with a checkpoint so the attempt is counted as a success
        let checkpoint_arc: Arc<Mutex<Option<GpuCheckpoint>>> =
            Arc::new(Mutex::new(Some(GpuCheckpoint::zeroed(4))));
        let recovery = DeviceLostRecovery::with_checkpoint(Arc::clone(&checkpoint_arc));

        let error = KwaversError::System(SystemError::ResourceUnavailable {
            resource: "GPU device_lost".to_string(),
        });
        let context =
            ErrorContext::new(crate::core::error::ErrorLocation::new("test.rs", 1, "test"));

        let _ = recovery.recover(&error, &context);

        let after = GpuRecoveryManager::global_stats();
        assert!(after.device_lost_attempts >= before.device_lost_attempts);
    }
}
