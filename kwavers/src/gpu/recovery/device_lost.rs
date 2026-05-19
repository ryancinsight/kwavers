use crate::core::error::recovery::{RecoveryResult, RecoveryStrategy};
use crate::core::error::{ErrorContext, KwaversError, KwaversResult, SystemError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::info;

use super::{update_avg_latency_us, GpuCheckpoint, GpuRecoveryAction, GLOBAL_STATS};

/// Device Lost Recovery Strategy
///
/// Handles GPU device loss by signalling the GPU time loop to re-initialize
/// the wgpu context and restore state from the last checkpoint.
///
/// **Target**: ≥99% success rate, <500ms latency
///
/// # Design
/// `recover()` does not perform async wgpu re-initialization itself (wgpu
/// adapter/device creation is async and requires a runtime context). Instead it
/// reads the last checkpoint from the shared mutex and returns
/// `GpuRecoveryAction::ReinitializeDevice { checkpoint }` so the time loop can
/// perform the actual re-init asynchronously.
#[derive(Debug)]
pub struct GpuDeviceLostRecovery {
    /// Success counter
    success_count: AtomicUsize,
    /// Total attempts counter
    total_count: AtomicUsize,
    /// Total latency counter (microseconds)
    total_latency_us: AtomicU64,
    /// Last-known good simulation state, updated by the GPU time loop.
    checkpoint: Arc<Mutex<Option<GpuCheckpoint>>>,
}

impl GpuDeviceLostRecovery {
    /// Create a new device-lost recovery strategy with no checkpoint.
    pub fn new() -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            checkpoint: Arc::new(Mutex::new(None)),
        }
    }

    /// Create a device-lost recovery strategy sharing an existing checkpoint arc.
    ///
    /// Used by [`GpuRecoveryManagerImpl`] so all strategies share the same checkpoint
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn with_zeroed_checkpoint(n_cells: usize) -> Self {
        let checkpoint = Arc::new(Mutex::new(Some(GpuCheckpoint::zeroed(n_cells))));
        Self::with_checkpoint(checkpoint)
    }

    /// Replace the currently stored checkpoint.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn set_checkpoint(&self, checkpoint: GpuCheckpoint) -> KwaversResult<()> {
        let mut guard = self.checkpoint.lock().map_err(|_| {
            KwaversError::InternalError("GpuDeviceLostRecovery: checkpoint mutex poisoned".to_string())
        })?;
        *guard = Some(checkpoint);
        Ok(())
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

    /// Check if device lost error
    fn is_device_lost(error: &KwaversError) -> bool {
        matches!(
            error,
            KwaversError::System(SystemError::ResourceUnavailable { resource })
            if resource.contains("GPU") && resource.contains("device_lost")
        )
    }
}

impl Default for GpuDeviceLostRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl RecoveryStrategy for GpuDeviceLostRecovery {
    fn recover(&self, error: &KwaversError, _context: &ErrorContext) -> RecoveryResult {
        let start = Instant::now();
        self.total_count.fetch_add(1, Ordering::Relaxed);

        // Verify this is a device lost error
        if !Self::is_device_lost(error) {
            return Err(KwaversError::InternalError(
                "GpuDeviceLostRecovery cannot handle non-device-lost errors".to_string(),
            ));
        }

        info!("Attempting device lost recovery...");

        // Read the last known checkpoint from the shared mutex.
        // If no checkpoint has been stored yet the recovery cannot proceed:
        // the time loop has not yet saved any state to roll back to.
        let checkpoint = match self.checkpoint.lock() {
            Ok(guard) => match guard.clone() {
                Some(cp) => cp,
                None => {
                    if let Ok(mut stats) = GLOBAL_STATS.lock() {
                        stats.device_lost_attempts += 1;
                    }
                    return Err(KwaversError::InternalError(
                        "GpuDeviceLostRecovery: no checkpoint available for device-lost recovery"
                            .to_string(),
                    ));
                }
            },
            Err(_) => {
                if let Ok(mut stats) = GLOBAL_STATS.lock() {
                    stats.device_lost_attempts += 1;
                }
                return Err(KwaversError::InternalError(
                    "GpuDeviceLostRecovery: checkpoint mutex poisoned".to_string(),
                ));
            }
        };

        let recovery_time = start.elapsed();
        let latency_us = recovery_time.as_micros() as u64;
        self.total_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut stats) = GLOBAL_STATS.lock() {
            stats.device_lost_attempts += 1;
            stats.device_lost_successes += 1;
            update_avg_latency_us(&mut stats, latency_us);
        }

        info!(
            step = checkpoint.step,
            latency_ms = latency_us as f64 / 1000.0,
            "Device lost recovery: signalling caller to reinitialize device from checkpoint"
        );

        Ok(Box::new(GpuRecoveryAction::ReinitializeDevice {
            checkpoint,
        }))
    }

    fn can_handle(&self, error: &KwaversError) -> bool {
        Self::is_device_lost(error)
    }

    fn strategy_name(&self) -> &'static str {
        "GpuDeviceLostRecovery"
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
    use crate::gpu::recovery::GpuRecoveryManagerImpl;

    #[test]
    fn device_lost_success_rate() {
        let recovery = GpuDeviceLostRecovery::new();

        // Initial rate should be 1.0 (no attempts)
        assert_eq!(recovery.success_rate(), 1.0);

        // Create a device lost error
        let error = KwaversError::System(SystemError::ResourceUnavailable {
            resource: "GPU device_lost".to_string(),
        });
        let context =
            ErrorContext::new(crate::core::error::ErrorLocation::new("test.rs", 1, "test"));

        // No checkpoint stored — recovery must return Err
        let result = recovery.recover(&error, &context);
        assert!(
            result.is_err(),
            "GpuDeviceLostRecovery with no checkpoint must return Err"
        );

        // Stats should be updated (attempt counted even on failure)
        let stats = GpuRecoveryManagerImpl::global_stats();
        assert!(stats.device_lost_attempts >= 1);
    }

    #[test]
    fn device_lost_recovery_with_checkpoint() {
        let checkpoint_arc: Arc<Mutex<Option<GpuCheckpoint>>> =
            Arc::new(Mutex::new(Some(GpuCheckpoint::zeroed(8))));
        let recovery = GpuDeviceLostRecovery::with_checkpoint(Arc::clone(&checkpoint_arc));

        let error = KwaversError::System(SystemError::ResourceUnavailable {
            resource: "GPU device_lost".to_string(),
        });
        let context =
            ErrorContext::new(crate::core::error::ErrorLocation::new("test.rs", 1, "test"));

        let result = recovery.recover(&error, &context);
        assert!(
            result.is_ok(),
            "GpuDeviceLostRecovery with checkpoint must succeed"
        );
    }
}
