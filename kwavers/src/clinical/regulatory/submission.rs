use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

use super::device::{DeviceDescription, PredicateDevice};
use super::evidence::ClinicalEvidence;
use super::iso8601_now;
use super::performance::PerformanceTest;
use super::risk::RiskRecord;
use std::time::{SystemTime, UNIX_EPOCH};

/// FDA 510(k) submission document
#[derive(Debug, Clone)]
pub struct SubmissionDocument {
    /// Submission ID
    pub submission_id: String,
    /// Manufacturer name
    pub manufacturer: String,
    /// Contact information
    pub contact_info: String,
    /// Device description
    pub device_description: DeviceDescription,
    /// Predicate device(s)
    pub predicate_devices: Vec<PredicateDevice>,
    /// Risk management data
    pub risk_management: Vec<RiskRecord>,
    /// Performance test data
    pub performance_tests: Vec<PerformanceTest>,
    /// Clinical evidence
    pub clinical_evidence: Vec<ClinicalEvidence>,
    /// General description statement
    pub summary: String,
    /// Submission date (ISO 8601)
    pub submission_date: String,
    /// Submission status: Draft, Submitted, Under Review, Approved, Rejected
    pub status: String,
}

impl SubmissionDocument {
    /// Create new 510(k) submission document
    pub fn new(
        manufacturer: impl Into<String>,
        device_description: DeviceDescription,
    ) -> KwaversResult<Self> {
        device_description.validate()?;

        let submission_id = format!(
            "SUBM_{:08}",
            (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                % 100000000)
        );

        Ok(Self {
            submission_id,
            manufacturer: manufacturer.into(),
            contact_info: String::new(),
            device_description,
            predicate_devices: Vec::new(),
            risk_management: Vec::new(),
            performance_tests: Vec::new(),
            clinical_evidence: Vec::new(),
            summary: String::new(),
            submission_date: iso8601_now(),
            status: "Draft".to_string(),
        })
    }

    /// Set contact information
    pub fn set_contact(&mut self, contact: impl Into<String>) {
        self.contact_info = contact.into();
    }

    /// Add predicate device
    pub fn add_predicate(&mut self, predicate: PredicateDevice) -> KwaversResult<()> {
        predicate.validate()?;
        self.predicate_devices.push(predicate);
        Ok(())
    }

    /// Add risk record
    pub fn add_risk(&mut self, risk: RiskRecord) {
        self.risk_management.push(risk);
    }

    /// Add performance test
    pub fn add_performance_test(&mut self, test: PerformanceTest) -> KwaversResult<()> {
        test.validate()?;
        self.performance_tests.push(test);
        Ok(())
    }

    /// Add clinical evidence
    pub fn add_clinical_evidence(&mut self, evidence: ClinicalEvidence) -> KwaversResult<()> {
        evidence.validate()?;
        self.clinical_evidence.push(evidence);
        Ok(())
    }

    /// Set summary statement
    pub fn set_summary(&mut self, summary: impl Into<String>) {
        self.summary = summary.into();
    }

    /// Generate submission checklist
    pub fn generate_checklist(&self) -> HashMap<String, bool> {
        let mut checklist = HashMap::new();

        checklist.insert("Device Description Complete".to_string(), {
            self.device_description.validate().is_ok()
        });

        checklist.insert("Predicate Device(s) Identified".to_string(), {
            !self.predicate_devices.is_empty()
        });

        checklist.insert("Risk Analysis Complete".to_string(), {
            !self.risk_management.is_empty()
        });

        checklist.insert("Performance Testing Done".to_string(), {
            self.performance_tests.iter().all(|t| t.result.is_some())
        });

        checklist.insert("Clinical Evidence Provided".to_string(), {
            !self.clinical_evidence.is_empty()
        });

        checklist.insert("Summary Statement Complete".to_string(), {
            !self.summary.is_empty()
        });

        checklist.insert("Contact Information Provided".to_string(), {
            !self.contact_info.is_empty()
        });

        checklist
    }

    /// Validate submission completeness
    pub fn validate(&self) -> KwaversResult<()> {
        self.device_description.validate()?;

        if self.predicate_devices.is_empty() {
            return Err(KwaversError::InvalidInput(
                "At least one predicate device required".to_string(),
            ));
        }

        for predicate in &self.predicate_devices {
            predicate.validate()?;
        }

        if self.performance_tests.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Performance testing data required".to_string(),
            ));
        }

        for test in &self.performance_tests {
            if test.result.is_none() {
                return Err(KwaversError::InvalidInput(format!(
                    "Test {} has no recorded result",
                    test.test_id
                )));
            }

            if !test.result.unwrap_or(false) {
                return Err(KwaversError::InvalidInput(format!(
                    "Test {} failed - cannot submit",
                    test.test_id
                )));
            }
        }

        if self.summary.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Summary statement required".to_string(),
            ));
        }

        Ok(())
    }

    /// Mark submission as submitted
    pub fn submit(&mut self) -> KwaversResult<()> {
        self.validate()?;
        self.status = "Submitted".to_string();
        Ok(())
    }

    /// Get submission statistics
    pub fn statistics(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();

        stats.insert("Submission ID".to_string(), self.submission_id.clone());

        stats.insert(
            "Device Name".to_string(),
            self.device_description.name.clone(),
        );

        stats.insert(
            "Classification".to_string(),
            self.device_description.classification.as_str().to_string(),
        );

        stats.insert(
            "Predicate Devices".to_string(),
            self.predicate_devices.len().to_string(),
        );

        stats.insert(
            "Risk Items".to_string(),
            self.risk_management.len().to_string(),
        );

        stats.insert(
            "Performance Tests".to_string(),
            self.performance_tests.len().to_string(),
        );

        let passed_tests = self
            .performance_tests
            .iter()
            .filter(|t| t.result == Some(true))
            .count();

        stats.insert(
            "Passed Tests".to_string(),
            format!("{}/{}", passed_tests, self.performance_tests.len()),
        );

        stats.insert(
            "Clinical Evidence Items".to_string(),
            self.clinical_evidence.len().to_string(),
        );

        stats.insert("Status".to_string(), self.status.clone());

        stats
    }
}
