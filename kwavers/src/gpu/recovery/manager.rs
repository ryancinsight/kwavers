use crate::core::error::recovery::{RecoveryResult, RecoveryStrategy};
use crate::core::error::{ErrorContext, KwaversError};
use std::sync::{Arc, Mutex};
use tracing::warn;

use super::{
    DeviceLostRecovery, GpuCheckpoint, GpuOomRecovery, GpuRecoveryStats, TimeoutRecovery,
    GLOBAL_STATS, RECOVERY_SUCCESS_THRESHOLD,
};

/// GPU Recovery Manager
///
/// Composite manager for GPU-specific recovery strategies.
/// Provides unified interface for GPU fault tolerance.
///
/// # Checkpoint sharing
/// All three strategies share the same `Arc<Mutex<Option<GpuCheckpoint>>>`.
/// The GPU time loop calls [`GpuRecoveryManager::update_checkpoint`] every
/// `checkpoint_interval` steps so that all strategies always have access to
/// the most recent consistent simulation state.
#[derive(Debug)]
pub struct GpuRecoveryManager {
    /// Device lost recovery
    device_lost: DeviceLostRecovery,
    /// OOM recovery
    oom: GpuOomRecovery,
    /// Timeout recovery
    timeout: TimeoutRecovery,
    /// Shared checkpoint updated by the GPU time loop.
    shared_checkpoint: Arc<Mutex<Option<GpuCheckpoint>>>,
}

impl Default for GpuRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuRecoveryManager {
    /// Create new GPU recovery manager with a shared checkpoint.
    pub fn new() -> Self {
        let shared_checkpoint: Arc<Mutex<Option<GpuCheckpoint>>> = Arc::new(Mutex::new(None));
        Self {
            device_lost: DeviceLostRecovery::with_checkpoint(Arc::clone(&shared_checkpoint)),
            oom: GpuOomRecovery::with_checkpoint(Arc::clone(&shared_checkpoint)),
            timeout: TimeoutRecovery::new(),
            shared_checkpoint,
        }
    }

    /// Store a new checkpoint that recovery strategies will use on the next fault.
    ///
    /// Called by the GPU time loop every `checkpoint_interval` steps after a
    /// successful GPU→CPU staging readback.
    pub fn update_checkpoint(&self, checkpoint: GpuCheckpoint) {
        if let Ok(mut guard) = self.shared_checkpoint.lock() {
            *guard = Some(checkpoint);
        } else {
            warn!("GpuRecoveryManager: checkpoint mutex poisoned; checkpoint not updated");
        }
    }

    /// Attempt recovery from GPU error
    pub fn recover_gpu_error(
        &self,
        error: &KwaversError,
        context: &ErrorContext,
    ) -> RecoveryResult {
        // Try strategies in order of specificity

        // 1. Device lost (most specific)
        if self.device_lost.can_handle(error) {
            return self.device_lost.recover(error, context);
        }

        // 2. OOM
        if self.oom.can_handle(error) {
            return self.oom.recover(error, context);
        }

        // 3. Timeout
        if self.timeout.can_handle(error) {
            return self.timeout.recover(error, context);
        }

        // No strategy can handle this error
        Err(KwaversError::InternalError(
            "No GPU recovery strategy can handle this error".to_string(),
        ))
    }

    /// Check if any strategy can handle this error
    pub fn can_recover(&self, error: &KwaversError) -> bool {
        self.device_lost.can_handle(error)
            || self.oom.can_handle(error)
            || self.timeout.can_handle(error)
    }

    /// Get global GPU recovery statistics
    pub fn global_stats() -> GpuRecoveryStats {
        GLOBAL_STATS
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    /// Get individual strategy rates
    pub fn strategy_rates(&self) -> StrategyRates {
        StrategyRates {
            device_lost: self.device_lost.success_rate(),
            oom: self.oom.success_rate(),
            timeout: self.timeout.success_rate(),
        }
    }

    /// Check if all strategies meet threshold
    pub fn meets_threshold(&self) -> bool {
        self.strategy_rates().meets_threshold()
    }
}

/// Strategy success rates
#[derive(Debug, Clone, Copy)]
pub struct StrategyRates {
    /// Device lost recovery rate
    pub device_lost: f64,
    /// OOM recovery rate
    pub oom: f64,
    /// Timeout recovery rate
    pub timeout: f64,
}

impl StrategyRates {
    /// Check if all rates meet threshold
    pub fn meets_threshold(&self) -> bool {
        self.device_lost >= RECOVERY_SUCCESS_THRESHOLD
            && self.oom >= RECOVERY_SUCCESS_THRESHOLD
            && self.timeout >= RECOVERY_SUCCESS_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SystemError;

    #[test]
    fn strategy_rates_threshold() {
        let rates = StrategyRates {
            device_lost: 0.99,
            oom: 0.95,
            timeout: 0.90,
        };
        assert!(rates.meets_threshold());

        let low_rates = StrategyRates {
            device_lost: 0.80,
            oom: 0.95,
            timeout: 0.90,
        };
        assert!(!low_rates.meets_threshold());
    }

    #[test]
    fn gpu_recovery_manager_creation() {
        let manager = GpuRecoveryManager::new();
        // No attempts yet — each strategy returns 1.0, so meets_threshold() is true
        assert!(manager.meets_threshold());

        let rates = manager.strategy_rates();
        assert_eq!(rates.device_lost, 1.0); // No attempts
    }

    #[test]
    fn gpu_recovery_manager_update_checkpoint() {
        let manager = GpuRecoveryManager::new();
        let cp = GpuCheckpoint::zeroed(16);
        manager.update_checkpoint(cp);

        // After updating the checkpoint both device-lost and OOM recovery succeed
        let dl_error = KwaversError::System(SystemError::ResourceUnavailable {
            resource: "GPU device_lost".to_string(),
        });
        let context =
            ErrorContext::new(crate::core::error::ErrorLocation::new("test.rs", 1, "test"));

        let result = manager.recover_gpu_error(&dl_error, &context);
        assert!(
            result.is_ok(),
            "device-lost recovery should succeed after checkpoint update"
        );
    }

    #[test]
    fn can_handle_detection() {
        let manager = GpuRecoveryManager::new();

        let device_lost = KwaversError::System(SystemError::ResourceUnavailable {
            resource: "GPU device_lost".to_string(),
        });
        assert!(manager.can_recover(&device_lost));

        let unrelated = KwaversError::InternalError("unrelated".to_string());
        assert!(!manager.can_recover(&unrelated));
    }
}
