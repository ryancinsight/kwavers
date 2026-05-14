//! GPU allocation configuration.

/// GPU Allocation Tracking Configuration.
#[derive(Debug, Clone)]
pub struct GpuAllocationConfig {
    /// Safety factor for memory budget (default 0.9 = 90%).
    pub safety_factor: f64,
    /// Enable error scope tracking.
    pub enable_error_scopes: bool,
    /// Log level for allocation events.
    pub log_level: tracing::Level,
}

impl Default for GpuAllocationConfig {
    fn default() -> Self {
        Self {
            safety_factor: 0.9,
            enable_error_scopes: true,
            log_level: tracing::Level::DEBUG,
        }
    }
}

impl GpuAllocationConfig {
    /// Create configuration with a custom safety factor clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn with_safety_factor(safety_factor: f64) -> Self {
        Self {
            safety_factor: safety_factor.clamp(0.0, 1.0),
            enable_error_scopes: true,
            log_level: tracing::Level::DEBUG,
        }
    }
}
