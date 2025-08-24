//! Validation reporting

use serde::{Deserialize, Serialize};

/// Validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Test name
    pub test_name: String,
    /// Test passed
    pub passed: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Validation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Tests failed
    pub tests_failed: usize,
    /// Overall pass rate
    pub pass_rate: f64,
}