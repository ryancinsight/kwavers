//! Validation configuration

use serde::{Deserialize, Serialize};

/// Configuration for validation tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum allowed error tolerance
    pub error_tolerance: f64,
    /// Number of test iterations
    pub num_iterations: usize,
    /// Enable convergence testing
    pub test_convergence: bool,
    /// Enable performance benchmarking
    pub benchmark_performance: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            error_tolerance: 1e-6,
            num_iterations: 100,
            test_convergence: true,
            benchmark_performance: false,
        }
    }
}