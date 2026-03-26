use super::iso8601_now;
use crate::core::error::{KwaversError, KwaversResult};

/// Performance test specification
#[derive(Debug, Clone)]
pub struct PerformanceTest {
    /// Test ID
    pub test_id: String,
    /// Test name
    pub name: String,
    /// Test objective
    pub objective: String,
    /// Test method/standard used
    pub method: String,
    /// Acceptance criteria
    pub acceptance_criteria: String,
    /// Test results (pass/fail)
    pub result: Option<bool>,
    /// Test data/findings
    pub findings: Vec<String>,
    /// Date conducted (ISO 8601)
    pub date_conducted: Option<String>,
}

impl PerformanceTest {
    /// Create new performance test
    pub fn new(test_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            test_id: test_id.into(),
            name: name.into(),
            objective: String::new(),
            method: String::new(),
            acceptance_criteria: String::new(),
            result: None,
            findings: Vec::new(),
            date_conducted: None,
        }
    }

    /// Set test objective
    pub fn with_objective(mut self, objective: impl Into<String>) -> Self {
        self.objective = objective.into();
        self
    }

    /// Set test method
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    /// Set acceptance criteria
    pub fn with_criteria(mut self, criteria: impl Into<String>) -> Self {
        self.acceptance_criteria = criteria.into();
        self
    }

    /// Record test result
    pub fn record_result(&mut self, passed: bool) {
        self.result = Some(passed);
        self.date_conducted = Some(iso8601_now());
    }

    /// Add test finding
    pub fn add_finding(&mut self, finding: impl Into<String>) {
        self.findings.push(finding.into());
    }

    /// Validate performance test
    pub fn validate(&self) -> KwaversResult<()> {
        if self.objective.is_empty() || self.method.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Test objective and method required".to_string(),
            ));
        }

        Ok(())
    }
}
