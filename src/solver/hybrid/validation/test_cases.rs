//! Validation test cases

use crate::error::KwaversResult;

/// Test result
#[derive(Debug, Clone))]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Test passed
    pub passed: bool,
    /// Error metrics
    pub error_metrics: Option<super::ErrorMetrics>,
    /// Performance metrics
    pub performance_metrics: Option<super::PerformanceMetrics>,
}

/// Trait for validation test cases
pub trait ValidationTestCase {
    /// Test name
    fn name(&self) -> &str;

    /// Run the test
    fn run(&self) -> KwaversResult<TestResult>;

    /// Check if results are within bounds
    fn check_bounds(&self, result: &TestResult, bounds: &super::ErrorBounds) -> bool;
}
