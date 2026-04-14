use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use tracing::warn;

use crate::core::error::{ErrorContext, KwaversError};

use super::super::{RecoveryAction, RecoveryResult, RecoveryStrategy};

/// Convergence failure recovery: switch solver implementation.
#[derive(Debug)]
pub struct ConvergenceFailureRecovery {
    max_switches: u32,
    switch_count: AtomicU32,
    success_rate: AtomicU64,
}

impl Default for ConvergenceFailureRecovery {
    fn default() -> Self {
        Self {
            max_switches: 3,
            switch_count: AtomicU32::new(0),
            success_rate: AtomicU64::new(0.85_f64.to_bits()),
        }
    }
}

impl ConvergenceFailureRecovery {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_max_switches(mut self, max: u32) -> Self {
        self.max_switches = max;
        self
    }

    #[must_use]
    pub fn switch_count(&self) -> u32 {
        self.switch_count.load(Ordering::Relaxed)
    }

    fn increment_switch_count(&self) {
        self.switch_count.fetch_add(1, Ordering::Relaxed);
    }
}

impl RecoveryStrategy for ConvergenceFailureRecovery {
    fn recover(&self, error: &KwaversError, _context: &ErrorContext) -> RecoveryResult {
        let current_count = self.switch_count();
        if current_count >= self.max_switches {
            return Err(KwaversError::InternalError(format!(
                "Maximum solver switches ({}) exceeded",
                self.max_switches
            )));
        }

        match error {
            KwaversError::Numerical(err) => {
                warn!(
                    numerical_error = %err,
                    switch_count = current_count,
                    "Convergence failure, switching solver algorithm"
                );
                self.increment_switch_count();
                Ok(Box::new(RecoveryAction::SwitchSolver {
                    target: "stable_fallback",
                }))
            }
            _ => Err(KwaversError::InternalError(
                "Convergence recovery called for non-numerical error".to_string(),
            )),
        }
    }

    fn can_handle(&self, error: &KwaversError) -> bool {
        matches!(error, KwaversError::Numerical(e) if e.to_string().contains("convergence") || e.to_string().contains("divergence"))
    }

    fn strategy_name(&self) -> &'static str {
        "ConvergenceFailureRecovery"
    }

    fn success_rate(&self) -> f64 {
        f64::from_bits(self.success_rate.load(Ordering::Relaxed))
    }
}
