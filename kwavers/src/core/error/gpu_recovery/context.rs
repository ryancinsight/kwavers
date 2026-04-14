use crate::core::error::context::{ErrorContext, ErrorLocation};
use crate::core::error::KwaversResult;
use crate::core::error::gpu::GpuError;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result type for recovery operations - boxed for object safety and dynamic dispatch
pub type RecoveryResult = KwaversResult<Box<dyn Any + Send>>;

/// Recovery context providing contextual information for GPU error recovery
#[derive(Debug, Clone)]
pub struct RecoveryContext {
    /// Error context with causal chain preserved
    pub error_context: ErrorContext,
    /// Timestamp when error occurred
    pub error_timestamp: Instant,
    /// GPU device reference (if available)
    pub device_id: Option<String>,
    /// Current simulation step
    pub current_step: usize,
    /// Pressure field snapshot for state preservation
    pub field_state: Option<Arc<ndarray::Array3<f64>>>,
}

impl RecoveryContext {
    /// Create new recovery context with error location
    pub fn new(location: ErrorLocation) -> Self {
        Self {
            error_context: ErrorContext::new(location),
            error_timestamp: Instant::now(),
            device_id: None,
            current_step: 0,
            field_state: None,
        }
    }

    /// Create context with current simulation step
    pub fn with_step(mut self, step: usize) -> Self {
        self.current_step = step;
        self
    }

    /// Create context with device ID
    pub fn with_device(mut self, device_id: impl Into<String>) -> Self {
        self.device_id = Some(device_id.into());
        self
    }

    /// Create context with field state preservation
    pub fn with_field_state(mut self, field: ndarray::Array3<f64>) -> Self {
        self.field_state = Some(Arc::new(field));
        self
    }

    /// Calculate elapsed time since error
    pub fn elapsed(&self) -> Duration {
        self.error_timestamp.elapsed()
    }
}

/// GPU-specific recovery strategy trait
pub trait GpuRecoveryStrategy: Debug + Send + Sync {
    /// Check if this strategy can handle the given GPU error
    fn can_handle(&self, error: &GpuError) -> bool;

    /// Attempt recovery from GPU error
    fn recover(&self, ctx: &RecoveryContext) -> RecoveryResult;

    /// Strategy identifier for telemetry and logging
    fn strategy_name(&self) -> &'static str;

    /// Current success rate from historical recovery attempts
    fn success_rate(&self) -> f64;

    /// Average recovery latency across all attempts
    fn avg_latency(&self) -> Duration;
}
