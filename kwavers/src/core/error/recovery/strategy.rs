use std::fmt::Debug;

use crate::core::error::{ErrorContext, KwaversError, KwaversResult};

/// Result type for recovery strategies - boxed for object safety and compatibility.
pub type RecoveryResult = KwaversResult<Box<dyn std::any::Any + Send>>;

/// Recovery strategy trait defining automated responses to known error classes.
pub trait RecoveryStrategy: Debug + Send + Sync {
    /// Attempt recovery from an error.
    fn recover(&self, error: &KwaversError, context: &ErrorContext) -> RecoveryResult;

    /// Check if this strategy can handle the given error.
    fn can_handle(&self, error: &KwaversError) -> bool;

    /// Strategy identifier for telemetry and logging.
    fn strategy_name(&self) -> &'static str;

    /// Historical success rate estimate.
    fn success_rate(&self) -> f64;
}

impl<T> RecoveryStrategy for Box<T>
where
    T: RecoveryStrategy + ?Sized,
{
    fn recover(&self, error: &KwaversError, context: &ErrorContext) -> RecoveryResult {
        (**self).recover(error, context)
    }

    fn can_handle(&self, error: &KwaversError) -> bool {
        (**self).can_handle(error)
    }

    fn strategy_name(&self) -> &'static str {
        (**self).strategy_name()
    }

    fn success_rate(&self) -> f64 {
        (**self).success_rate()
    }
}
