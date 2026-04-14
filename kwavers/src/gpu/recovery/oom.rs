use crate::core::error::recovery::{RecoveryResult, RecoveryStrategy};
use crate::core::error::{ErrorContext, KwaversError, KwaversResult, SystemError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::info;

use super::{update_avg_latency_us, GpuCheckpoint, GpuRecoveryAction, GLOBAL_STATS};

/// GPU OOM Recovery Strategy with CPU Fallback
///
/// Handles GPU out-of-memory by signalling the GPU time loop to fall back to
/// the CPU solver while preserving simulation state via the last checkpoint.
///
/// **Target**: ≥95% success rate, <100ms latency
///
/// # Design
/// `recover()` does not migrate solver state itself — it cannot hold a reference
/// to the solver without introducing a circular dependency. Instead it reads the
/// last checkpoint and returns `GpuRecoveryAction::FallbackToCpu { checkpoint }`
/// so the time loop can construct a CPU solver and resume.
#[derive(Debug)]
pub struct GpuOomRecovery {
    /// Success counter
    success_count: AtomicUsize,
    /// Total attempts counter
    total_count: AtomicUsize,
    /// Total latency counter (microseconds)
    total_latency_us: AtomicU64,
    /// Last-known good simulation state, updated by the GPU time loop.
    checkpoint: Arc<Mutex<Option<GpuCheckpoint>>>,
}

impl GpuOomRecovery {
    /// Create a new GPU OOM recovery strategy with no checkpoint.
    pub fn new() -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            checkpoint: Arc::new(Mutex::new(None)),
        }
    }

    /// Create an OOM recovery strategy sharing an existing checkpoint arc.
    ///
    /// Used by [`GpuRecoveryManager`] so all strategies share the same checkpoint
    /// written by the GPU time loop.
    pub fn with_checkpoint(checkpoint: Arc<Mutex<Option<GpuCheckpoint>>>) -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            checkpoint,
        }
    }

    /// Create a recovery strategy with a pre-seeded zeroed checkpoint.
    pub fn with_zeroed_checkpoint(n_cells: usize) -> Self {
        let checkpoint = Arc::new(Mutex::new(Some(GpuCheckpoint::zeroed(n_cells))));
        Self::with_checkpoint(checkpoint)
    }

    /// Replace the currently stored checkpoint.
    pub fn set_checkpoint(&self, checkpoint: GpuCheckpoint) -> KwaversResult<()> {
        let mut guard = self.checkpoint.lock().map_err(|_| {
            KwaversError::InternalError("GpuOomRecovery: checkpoint mutex poisoned".to_string())
        })?;
        *guard = Some(checkpoint);
        Ok(())
    }

    /// Check if OOM error
    fn is_oom(error: &KwaversError) -> bool {
        if let KwaversError::System(SystemError::ResourceExhausted { resource, .. }) = error {
            resource.contains("GPU") && resource.contains("memory")
        } else {
            false
        }
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

impl Default for GpuOomRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl RecoveryStrategy for GpuOomRecovery {
    fn recover(&self, error: &KwaversError, _context: &ErrorContext) -> RecoveryResult {
        let start = Instant::now();
        self.total_count.fetch_add(1, Ordering::Relaxed);

        // Verify this is an OOM error
        if !Self::is_oom(error) {
            return Err(KwaversError::InternalError(
                "GpuOomRecovery cannot handle non-OOM errors".to_string(),
            ));
        }

        info!("Attempting GPU OOM recovery with CPU fallback...");

        // Read the last known checkpoint from the shared mutex.
        // If no checkpoint has been stored yet the recovery cannot proceed.
        let checkpoint = match self.checkpoint.lock() {
            Ok(guard) => match guard.clone() {
                Some(cp) => cp,
                None => {
                    if let Ok(mut stats) = GLOBAL_STATS.lock() {
                        stats.oom_attempts += 1;
                    }
                    return Err(KwaversError::InternalError(
                        "GpuOomRecovery: no checkpoint available for CPU fallback".to_string(),
                    ));
                }
            },
            Err(_) => {
                if let Ok(mut stats) = GLOBAL_STATS.lock() {
                    stats.oom_attempts += 1;
                }
                return Err(KwaversError::InternalError(
                    "GpuOomRecovery: checkpoint mutex poisoned".to_string(),
                ));
            }
        };

        let recovery_time = start.elapsed();
        let latency_us = recovery_time.as_micros() as u64;
        self.total_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut stats) = GLOBAL_STATS.lock() {
            stats.oom_attempts += 1;
            stats.oom_successes += 1;
            update_avg_latency_us(&mut stats, latency_us);
        }

        info!(
            step = checkpoint.step,
            latency_ms = latency_us as f64 / 1000.0,
            "GPU OOM recovery: signalling caller to fall back to CPU solver from checkpoint"
        );

        Ok(Box::new(GpuRecoveryAction::FallbackToCpu { checkpoint }))
    }

    fn can_handle(&self, error: &KwaversError) -> bool {
        Self::is_oom(error)
    }

    fn strategy_name(&self) -> &'static str {
        "GpuOomRecovery"
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
    fn oom_recovery_rate() {
        let recovery = GpuOomRecovery::new();
        // When no attempts have been made, success_rate returns 1.0
        assert_eq!(recovery.success_rate(), 1.0);

        // Simulate OOM error
        let error = KwaversError::System(SystemError::ResourceExhausted {
            resource: "GPU memory".to_string(),
            reason: "OOM".to_string(),
        });

        assert!(GpuOomRecovery::is_oom(&error));
    }

    #[test]
    fn oom_recovery_with_checkpoint() {
        let checkpoint_arc: Arc<Mutex<Option<GpuCheckpoint>>> =
            Arc::new(Mutex::new(Some(GpuCheckpoint::zeroed(8))));
        let recovery = GpuOomRecovery::with_checkpoint(Arc::clone(&checkpoint_arc));

        let error = KwaversError::System(SystemError::ResourceExhausted {
            resource: "GPU memory".to_string(),
            reason: "OOM".to_string(),
        });
        let context =
            ErrorContext::new(crate::core::error::ErrorLocation::new("test.rs", 1, "test"));

        let result = recovery.recover(&error, &context);
        assert!(
            result.is_ok(),
            "GpuOomRecovery with checkpoint must succeed"
        );
    }
}
