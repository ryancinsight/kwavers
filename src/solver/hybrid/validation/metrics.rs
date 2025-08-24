//! Validation metrics

use serde::{Deserialize, Serialize};

/// Error metrics for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// L2 norm error
    pub l2_error: f64,
    /// L∞ norm error
    pub linf_error: f64,
    /// Relative error
    pub relative_error: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Computation time in seconds
    pub computation_time: f64,
    /// Memory usage in MB
    pub memory_usage: f64,
    /// Throughput in points/second
    pub throughput: f64,
}

/// Error bounds for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBounds {
    /// Maximum allowed L2 error
    pub max_l2_error: f64,
    /// Maximum allowed L∞ error
    pub max_linf_error: f64,
    /// Maximum allowed relative error
    pub max_relative_error: f64,
}

impl Default for ErrorBounds {
    fn default() -> Self {
        Self {
            max_l2_error: 1e-6,
            max_linf_error: 1e-5,
            max_relative_error: 1e-4,
        }
    }
}