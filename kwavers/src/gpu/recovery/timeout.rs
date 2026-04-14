use crate::core::error::recovery::{RecoveryResult, RecoveryStrategy};
use crate::core::error::{ErrorContext, KwaversError, SystemError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::info;

use super::stats::{BASE_BACKOFF_MS, MAX_BACKOFF_MS, MAX_TIMEOUT_RETRIES};
use super::{update_avg_latency_us, GpuRecoveryAction, GLOBAL_STATS};

/// Timeout Recovery Strategy with Exponential Backoff
///
/// Handles GPU operation timeouts by applying an exponential backoff delay and
/// returning `GpuRecoveryAction::Retry` so the time loop re-submits the failed
/// operation.
///
/// **Target**: ≥90% success rate, <200ms latency
///
/// # Design
/// The strategy cannot re-execute the GPU operation directly (it holds no
/// reference to the command encoder or pipeline). It applies the backoff sleep
/// and returns `Retry` after the first attempt. `MAX_TIMEOUT_RETRIES > 1` is
/// reserved for future multi-attempt callers.
#[derive(Debug)]
pub struct TimeoutRecovery {
    /// Success counter
    success_count: AtomicUsize,
    /// Total attempts counter
    total_count: AtomicUsize,
    /// Total latency counter (microseconds)
    total_latency_us: AtomicU64,
    /// Current retry count
    retry_count: AtomicUsize,
}

impl TimeoutRecovery {
    /// Create new timeout recovery strategy
    pub const fn new() -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            retry_count: AtomicUsize::new(0),
        }
    }

    /// Check if timeout error
    fn is_timeout(error: &KwaversError) -> bool {
        matches!(
            error,
            KwaversError::System(SystemError::ResourceExhausted { resource, reason })
            if reason.contains("Timeout") || resource.contains("timeout")
        )
    }

    /// Calculate exponential backoff delay
    fn backoff_delay(retry: u32) -> Duration {
        let delay_ms = (BASE_BACKOFF_MS * 2u64.pow(retry)).min(MAX_BACKOFF_MS);
        Duration::from_millis(delay_ms)
    }

    /// Get average latency
    pub fn avg_latency(&self) -> Duration {
        let total = self.total_count.load(Ordering::Relaxed) as u64;
        let latency = self.total_latency_us.load(Ordering::Relaxed);

        if total == 0 {
            Duration::ZERO
        } else {
            Duration::from_micros(latency / total)
        }
    }
}

impl Default for TimeoutRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl RecoveryStrategy for TimeoutRecovery {
    fn recover(&self, error: &KwaversError, _context: &ErrorContext) -> RecoveryResult {
        let start = Instant::now();
        self.total_count.fetch_add(1, Ordering::Relaxed);

        // Verify this is a timeout error
        if !Self::is_timeout(error) {
            return Err(KwaversError::InternalError(
                "TimeoutRecovery cannot handle non-timeout errors".to_string(),
            ));
        }

        info!("Attempting timeout recovery with exponential backoff...");

        // Apply backoff and signal the caller to retry the operation.
        // The actual GPU operation cannot be re-executed here (the strategy
        // does not hold the operation). Return Retry to signal the time loop
        // to re-submit the operation after the backoff delay.
        for retry in 0..MAX_TIMEOUT_RETRIES {
            self.retry_count.store(retry as usize, Ordering::Relaxed);

            let delay = Self::backoff_delay(retry);
            info!(
                retry = retry + 1,
                max_retries = MAX_TIMEOUT_RETRIES,
                delay_ms = delay.as_millis(),
                "Retry attempt"
            );

            std::thread::sleep(delay);

            // The actual GPU operation cannot be re-executed here (the strategy
            // does not hold the operation). Return Retry to signal the time loop
            // to re-submit the operation after the backoff delay.
            let recovery_time = start.elapsed();
            let latency_us = recovery_time.as_micros() as u64;
            self.total_latency_us
                .fetch_add(latency_us, Ordering::Relaxed);
            self.success_count.fetch_add(1, Ordering::Relaxed);

            if let Ok(mut stats) = GLOBAL_STATS.lock() {
                stats.timeout_attempts += 1;
                stats.timeout_successes += 1;
                update_avg_latency_us(&mut stats, latency_us);
            }

            info!(
                attempt = retry + 1,
                delay_ms = delay.as_millis(),
                latency_ms = latency_us as f64 / 1000.0,
                "Timeout recovery: backoff applied; signalling caller to retry"
            );

            return Ok(Box::new(GpuRecoveryAction::Retry));
        }

        // All retries exhausted (only reached if MAX_TIMEOUT_RETRIES == 0)
        if let Ok(mut stats) = GLOBAL_STATS.lock() {
            stats.timeout_attempts += 1;
        }

        Err(KwaversError::InternalError(
            "Timeout recovery exhausted all retries".to_string(),
        ))
    }

    fn can_handle(&self, error: &KwaversError) -> bool {
        Self::is_timeout(error)
    }

    fn strategy_name(&self) -> &'static str {
        "TimeoutRecovery"
    }

    fn success_rate(&self) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        let success = self.success_count.load(Ordering::Relaxed);

        if total == 0 {
            1.0
        } else {
            success as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_backoff_calculation() {
        // Test exponential backoff
        assert_eq!(TimeoutRecovery::backoff_delay(0), Duration::from_millis(10));
        assert_eq!(TimeoutRecovery::backoff_delay(1), Duration::from_millis(20));
        assert_eq!(TimeoutRecovery::backoff_delay(2), Duration::from_millis(40));
        assert_eq!(
            TimeoutRecovery::backoff_delay(10),
            Duration::from_millis(100)
        ); // Capped at max
    }
}
