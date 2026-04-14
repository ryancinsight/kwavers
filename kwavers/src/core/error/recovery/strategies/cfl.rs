use std::sync::atomic::{AtomicU64, Ordering};

use tracing::warn;

use crate::core::error::{ErrorContext, KwaversError};

use super::super::{RecoveryAction, RecoveryResult, RecoveryStrategy};

const MIN_CFL_REDUCTION: f64 = 0.5;
const MAX_CFL_REDUCTION: f64 = 0.125;

/// CFL violation recovery: reduce timestep.
#[derive(Debug)]
pub struct CflViolationRecovery {
    reduction_factor: AtomicU64,
    success_rate: AtomicU64,
}

impl Default for CflViolationRecovery {
    fn default() -> Self {
        Self {
            reduction_factor: AtomicU64::new(0.5_f64.to_bits()),
            success_rate: AtomicU64::new(0.95_f64.to_bits()),
        }
    }
}

impl CflViolationRecovery {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn reduction_factor(&self) -> f64 {
        f64::from_bits(self.reduction_factor.load(Ordering::Relaxed))
    }

    pub fn set_reduction_factor(&self, factor: f64) {
        let clamped = factor.clamp(MAX_CFL_REDUCTION, MIN_CFL_REDUCTION);
        self.reduction_factor
            .store(clamped.to_bits(), Ordering::Relaxed);
    }
}

impl RecoveryStrategy for CflViolationRecovery {
    fn recover(&self, error: &KwaversError, _context: &ErrorContext) -> RecoveryResult {
        match error {
            KwaversError::Numerical(err) => {
                warn!(
                    numerical_error = %err,
                    reduction_factor = self.reduction_factor(),
                    "CFL violation detected, reducing timestep"
                );
                Ok(Box::new(RecoveryAction::ReduceTimestep {
                    factor: self.reduction_factor(),
                }))
            }
            _ => Err(KwaversError::InternalError(
                "CFL recovery called for non-numerical error".to_string(),
            )),
        }
    }

    fn can_handle(&self, error: &KwaversError) -> bool {
        matches!(error, KwaversError::Numerical(e) if e.to_string().contains("CFL") || e.to_string().contains("timestep"))
    }

    fn strategy_name(&self) -> &'static str {
        "CflViolationRecovery"
    }

    fn success_rate(&self) -> f64 {
        f64::from_bits(self.success_rate.load(Ordering::Relaxed))
    }
}
