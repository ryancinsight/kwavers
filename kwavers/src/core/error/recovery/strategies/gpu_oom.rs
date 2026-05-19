use std::sync::atomic::{AtomicU64, Ordering};

use tracing::info;

use crate::core::error::{ErrorContext, KwaversError};

use super::super::{RecoveryAction, RecoveryResult, RecoveryStrategy};

/// GPU OOM recovery: clear caches and fall back to CPU.
#[derive(Debug)]
pub struct ErrorRecoveryGpuOom {
    success_rate: AtomicU64,
}

impl Default for ErrorRecoveryGpuOom {
    fn default() -> Self {
        Self {
            success_rate: AtomicU64::new(0.9_f64.to_bits()),
        }
    }
}

impl ErrorRecoveryGpuOom {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl RecoveryStrategy for ErrorRecoveryGpuOom {
    fn recover(&self, _error: &KwaversError, _context: &ErrorContext) -> RecoveryResult {
        info!("Attempting GPU OOM recovery: clearing caches and falling back to CPU");
        Ok(Box::new(RecoveryAction::CpuFallback))
    }

    fn can_handle(&self, error: &KwaversError) -> bool {
        matches!(error, KwaversError::ResourceLimitExceeded { .. })
            || matches!(error, KwaversError::GpuError(message) if message.contains("OOM") || message.contains("out of memory"))
    }

    fn strategy_name(&self) -> &'static str {
        "ErrorRecoveryGpuOom"
    }

    fn success_rate(&self) -> f64 {
        f64::from_bits(self.success_rate.load(Ordering::Relaxed))
    }
}
