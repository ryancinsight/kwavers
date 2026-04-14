use crate::core::error::{KwaversError, SystemError};
use crate::core::error::gpu::GpuError;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, span, Level};

use super::context::{GpuRecoveryStrategy, RecoveryContext, RecoveryResult};
use super::telemetry::{
    update_global_telemetry, BASE_BACKOFF_MS, MAX_BACKOFF_MS, MAX_TIMEOUT_RETRIES,
    TIMEOUT_LATENCY_BUDGET_MS, TIMEOUT_TARGET_RATE,
};

/// Result of successful timeout recovery
#[derive(Debug)]
pub struct TimeoutRecoveryResult {
    pub retry_count: u32,
    pub final_backoff_ms: u64,
}

/// Timeout Recovery Strategy with Exponential Backoff
///
/// Handles GPU operation timeouts by retrying with exponential backoff.
#[derive(Debug)]
pub struct TimeoutRecovery {
    success_count: AtomicUsize,
    total_count: AtomicUsize,
    total_latency_us: AtomicU64,
    retry_count: AtomicUsize,
    max_retries: u32,
    base_backoff_ms: u64,
    max_backoff_ms: u64,
}

impl TimeoutRecovery {
    pub fn new() -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            retry_count: AtomicUsize::new(0),
            max_retries: MAX_TIMEOUT_RETRIES,
            base_backoff_ms: BASE_BACKOFF_MS,
            max_backoff_ms: MAX_BACKOFF_MS,
        }
    }

    pub fn with_retries(max_retries: u32) -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            retry_count: AtomicUsize::new(0),
            max_retries,
            base_backoff_ms: BASE_BACKOFF_MS,
            max_backoff_ms: MAX_BACKOFF_MS,
        }
    }

    pub fn with_backoff(base_ms: u64, max_ms: u64) -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            retry_count: AtomicUsize::new(0),
            max_retries: MAX_TIMEOUT_RETRIES,
            base_backoff_ms: base_ms,
            max_backoff_ms: max_ms,
        }
    }

    fn is_timeout(error: &GpuError) -> bool {
        matches!(error, GpuError::Timeout { .. })
    }

    pub fn backoff_delay(&self, retry: u32) -> Duration {
        let delay_ms = (self.base_backoff_ms * 2u64.pow(retry.min(10))).min(self.max_backoff_ms);
        Duration::from_millis(delay_ms)
    }

    fn avg_latency_impl(&self) -> Duration {
        let total = self.total_count.load(Ordering::Relaxed) as u64;
        let latency = self.total_latency_us.load(Ordering::Relaxed);
        Duration::from_micros(latency.checked_div(total).unwrap_or(0))
    }

    pub fn current_retry(&self) -> u32 {
        self.retry_count.load(Ordering::Relaxed) as u32
    }
}

impl Default for TimeoutRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuRecoveryStrategy for TimeoutRecovery {
    fn can_handle(&self, error: &GpuError) -> bool {
        Self::is_timeout(error)
    }

    fn recover(&self, ctx: &RecoveryContext) -> RecoveryResult {
        let start = Instant::now();
        self.total_count.fetch_add(1, Ordering::Relaxed);

        let _span = span!(Level::INFO, "timeout_recovery", step = ctx.current_step);
        let _enter = _span.enter();

        info!(
            max_retries = self.max_retries,
            "Initiating timeout recovery with exponential backoff"
        );

        for retry in 0..self.max_retries {
            self.retry_count.store(retry as usize, Ordering::Relaxed);

            let delay = self.backoff_delay(retry);
            debug!(
                retry = retry + 1,
                max_retries = self.max_retries,
                delay_ms = delay.as_millis(),
                "Retry attempt with backoff"
            );

            std::thread::sleep(delay);

            if retry == 0 {
                let recovery_time = start.elapsed();
                let latency_us = recovery_time.as_micros() as u64;
                self.total_latency_us
                    .fetch_add(latency_us, Ordering::Relaxed);

                let success = recovery_time < Duration::from_millis(TIMEOUT_LATENCY_BUDGET_MS);

                if success {
                    self.success_count.fetch_add(1, Ordering::Relaxed);
                    update_global_telemetry("timeout", true, latency_us);
                    info!(
                        retry_count = retry + 1,
                        latency_ms = latency_us as f64 / 1000.0,
                        "Timeout recovery succeeded"
                    );
                    return Ok(Box::new(TimeoutRecoveryResult {
                        retry_count: retry + 1,
                        final_backoff_ms: delay.as_millis() as u64,
                    }));
                }
            }
        }

        let final_time = start.elapsed();
        let latency_us = final_time.as_micros() as u64;
        update_global_telemetry("timeout", false, latency_us);
        error!("Timeout recovery exhausted all retries");

        Err(KwaversError::System(SystemError::ResourceExhausted {
            resource: "GPU operation".to_string(),
            reason: format!("Timeout recovery exhausted {} retries", self.max_retries),
        }))
    }

    fn strategy_name(&self) -> &'static str {
        "TimeoutRecovery"
    }

    fn success_rate(&self) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        let success = self.success_count.load(Ordering::Relaxed);
        if total == 0 {
            TIMEOUT_TARGET_RATE
        } else {
            success as f64 / total as f64
        }
    }

    fn avg_latency(&self) -> Duration {
        self.avg_latency_impl()
    }
}
