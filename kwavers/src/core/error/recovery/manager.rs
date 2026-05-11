use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tracing::{info, warn};

use crate::core::error::{ErrorContext, KwaversError};

use super::action::RecoveryAction;
use super::attempt::RecoveryAttempt;
use super::strategies::{CflViolationRecovery, ConvergenceFailureRecovery, GpuOomRecovery};
use super::strategy::{RecoveryResult, RecoveryStrategy};

/// Maximum number of recovery attempts before giving up.
const MAX_RECOVERY_ATTEMPTS: u32 = 3;
/// Threshold for considering recovery successful.
const RECOVERY_SUCCESS_THRESHOLD: f64 = 0.90;

/// Composite recovery manager with multiple strategies.
#[derive(Debug, Default)]
pub struct RecoveryManager {
    strategies: Vec<Arc<dyn RecoveryStrategy>>,
    attempt_history: Vec<RecoveryAttempt>,
}

impl RecoveryManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            attempt_history: Vec::new(),
        }
    }

    /// Register a boxed strategy.
    pub fn register_boxed(&mut self, strategy: Box<dyn RecoveryStrategy>) {
        self.strategies.push(Arc::from(strategy));
    }

    /// Register a strategy in evaluation order.
    pub fn register<S: RecoveryStrategy + 'static>(&mut self, strategy: S) {
        self.strategies.push(Arc::new(strategy));
    }

    /// Compatibility entrypoint preserving boxed payload behavior.
    pub fn recover(&mut self, error: &KwaversError, context: &ErrorContext) -> RecoveryResult {
        let action = self.recover_action(error, context)?;
        Ok(Box::new(action))
    }

    /// Backward-compatible alias.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn attempt_recovery(
        &mut self,
        error: &KwaversError,
        context: &ErrorContext,
    ) -> RecoveryResult {
        self.recover(error, context)
    }

    /// Typed recovery entrypoint for new call sites.
    /// # Errors
    /// - Returns [`KwaversError::InternalError`] if the precondition for a InternalError-class constraint is violated.
    ///
    pub fn recover_action(
        &mut self,
        error: &KwaversError,
        context: &ErrorContext,
    ) -> Result<RecoveryAction, KwaversError> {
        let start_time = Instant::now();
        let mut attempts_made = 0u32;

        for strategy in &self.strategies {
            if attempts_made >= MAX_RECOVERY_ATTEMPTS {
                warn!(
                    limit = MAX_RECOVERY_ATTEMPTS,
                    "Recovery attempt limit reached; giving up"
                );
                break;
            }

            if !strategy.can_handle(error) {
                continue;
            }

            attempts_made += 1;
            let mut attempt = RecoveryAttempt::new(strategy.strategy_name(), error);

            match strategy.recover(error, context) {
                Ok(value) => {
                    let duration = start_time.elapsed();
                    attempt.mark_succeeded();
                    attempt.set_duration(duration);
                    self.attempt_history.push(attempt);
                    info!(
                        strategy = strategy.strategy_name(),
                        duration_ms = duration.as_millis(),
                        "Recovery succeeded"
                    );
                    return decode_action(value);
                }
                Err(recovery_error) => {
                    attempt.mark_failed(&recovery_error);
                    attempt.set_duration(start_time.elapsed());
                    self.attempt_history.push(attempt);
                    warn!(
                        strategy = strategy.strategy_name(),
                        error = %recovery_error,
                        "Recovery attempt failed, trying next strategy"
                    );
                }
            }
        }

        Err(KwaversError::InternalError(format!(
            "All recovery strategies exhausted for error: {}",
            error
        )))
    }

    #[must_use]
    pub fn can_recover(&self, error: &KwaversError) -> bool {
        self.strategies
            .iter()
            .any(|strategy| strategy.can_handle(error))
    }

    #[must_use]
    pub fn statistics(&self) -> RecoveryStatistics {
        let total = self.attempt_history.len();
        let successful = self
            .attempt_history
            .iter()
            .filter(|attempt| attempt.succeeded)
            .count();

        let by_strategy = self.attempt_history.iter().fold(
            HashMap::<String, (usize, usize)>::new(),
            |mut acc, attempt| {
                let entry = acc.entry(attempt.strategy.clone()).or_insert((0, 0));
                entry.0 += 1;
                if attempt.succeeded {
                    entry.1 += 1;
                }
                acc
            },
        );

        RecoveryStatistics {
            total_attempts: total,
            successful_attempts: successful,
            overall_success_rate: if total > 0 {
                successful as f64 / total as f64
            } else {
                1.0
            },
            by_strategy,
        }
    }
}

fn decode_action(payload: Box<dyn std::any::Any + Send>) -> Result<RecoveryAction, KwaversError> {
    match payload.downcast::<RecoveryAction>() {
        Ok(action) => Ok(*action),
        Err(_) => Err(KwaversError::InternalError(
            "Recovery strategy returned a non-RecoveryAction payload".to_string(),
        )),
    }
}

/// Recovery statistics for monitoring and alerting.
#[derive(Debug, Clone)]
pub struct RecoveryStatistics {
    pub total_attempts: usize,
    pub successful_attempts: usize,
    pub overall_success_rate: f64,
    pub by_strategy: HashMap<String, (usize, usize)>,
}

impl RecoveryStatistics {
    #[must_use]
    pub fn meets_threshold(&self) -> bool {
        self.overall_success_rate >= RECOVERY_SUCCESS_THRESHOLD
    }

    #[must_use]
    pub fn strategy_success_rate(&self, strategy: &str) -> Option<f64> {
        self.by_strategy.get(strategy).map(|(total, success)| {
            if *total > 0 {
                *success as f64 / *total as f64
            } else {
                1.0
            }
        })
    }
}

/// Builder for constructing a manager with common strategies.
#[derive(Debug)]
pub struct RecoveryBuilder {
    manager: RecoveryManager,
}

impl RecoveryBuilder {
    #[must_use]
    pub fn new() -> Self {
        let mut manager = RecoveryManager::new();
        manager.register(GpuOomRecovery::new());
        manager.register(CflViolationRecovery::new());
        manager.register(ConvergenceFailureRecovery::new());
        Self { manager }
    }

    #[must_use]
    pub fn with_strategy<S: RecoveryStrategy + 'static>(mut self, strategy: S) -> Self {
        self.manager.register(strategy);
        self
    }

    #[must_use]
    pub fn build(self) -> RecoveryManager {
        self.manager
    }
}

impl Default for RecoveryBuilder {
    fn default() -> Self {
        Self::new()
    }
}
