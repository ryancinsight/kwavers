use std::time::{Duration, Instant};

use crate::core::error::KwaversError;

/// Recovery attempt metadata for causal chain preservation.
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    /// Timestamp of recovery attempt.
    pub timestamp: Instant,
    /// Strategy that attempted recovery.
    pub strategy: String,
    /// Whether recovery succeeded.
    pub succeeded: bool,
    /// Duration of recovery attempt.
    pub duration: Duration,
    /// Error that triggered recovery.
    pub original_error: String,
    /// Resulting error if recovery failed.
    pub recovery_error: Option<String>,
}

impl RecoveryAttempt {
    #[must_use]
    pub fn new(strategy: &str, original: &KwaversError) -> Self {
        Self {
            timestamp: Instant::now(),
            strategy: strategy.to_string(),
            succeeded: false,
            duration: Duration::ZERO,
            original_error: original.to_string(),
            recovery_error: None,
        }
    }

    pub fn mark_succeeded(&mut self) {
        self.succeeded = true;
    }

    pub fn mark_failed(&mut self, error: &KwaversError) {
        self.succeeded = false;
        self.recovery_error = Some(error.to_string());
    }

    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }
}
