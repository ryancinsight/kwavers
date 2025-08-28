//! Validation report generation

use std::fmt;

/// Test result for a single test case
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub error: f64,
    pub tolerance: f64,
    pub message: String,
}

/// Validation report containing all test results
#[derive(Debug, Default)]
pub struct ValidationReport {
    results: Vec<TestResult>,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a test result
    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    /// Get total number of tests
    pub fn total_tests(&self) -> usize {
        self.results.len()
    }

    /// Get number of passed tests
    pub fn passed_tests(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Get number of failed tests
    pub fn failed_tests(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Get pass rate as percentage
    pub fn pass_rate(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            100.0 * self.passed_tests() as f64 / self.total_tests() as f64
        }
    }

    /// Get maximum error across all tests
    pub fn max_error(&self) -> f64 {
        self.results.iter().map(|r| r.error).fold(0.0, f64::max)
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "k-Wave Validation Report")?;
        writeln!(f, "========================")?;
        writeln!(f, "Total Tests: {}", self.total_tests())?;
        writeln!(f, "Passed: {}", self.passed_tests())?;
        writeln!(f, "Failed: {}", self.failed_tests())?;
        writeln!(f, "Pass Rate: {:.1}%", self.pass_rate())?;
        writeln!(f, "Max Error: {:.2e}", self.max_error())?;

        if !self.results.is_empty() {
            writeln!(f, "\nTest Results:")?;
            for result in &self.results {
                let status = if result.passed { "✓" } else { "✗" };
                writeln!(
                    f,
                    "  {} {} - Error: {:.2e} (Tol: {:.2e})",
                    status, result.name, result.error, result.tolerance
                )?;
            }
        }

        Ok(())
    }
}
