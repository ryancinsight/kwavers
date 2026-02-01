//! Regulatory Documentation and FDA 510(k) Compliance
//!
//! Comprehensive regulatory documentation framework for FDA 510(k) medical device submissions.
//! Implements documentation generation, compliance tracking, and validation for ultrasound therapy devices.
//!
//! This module provides:
//! - Device description and intended use documentation
//! - Substantial equivalence and predicate device information
//! - Risk analysis and mitigation strategies
//! - Performance testing and validation reports
//! - Clinical evidence and safety data compilation
//! - Regulatory submission document generation

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Device classification per FDA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    /// Class I: Low risk, general controls
    ClassI,
    /// Class II: Moderate risk, requires 510(k) submission
    ClassII,
    /// Class III: High risk, requires Premarket Approval (PMA)
    ClassIII,
}

impl DeviceClass {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ClassI => "Class I",
            Self::ClassII => "Class II",
            Self::ClassIII => "Class III",
        }
    }

    /// Get risk level description
    pub fn risk_level(&self) -> &'static str {
        match self {
            Self::ClassI => "Low Risk",
            Self::ClassII => "Moderate Risk",
            Self::ClassIII => "High Risk",
        }
    }
}

/// Device description section
#[derive(Debug, Clone)]
pub struct DeviceDescription {
    /// Device name and model
    pub name: String,
    /// Device classification
    pub classification: DeviceClass,
    /// Intended use statement
    pub intended_use: String,
    /// Indications for use
    pub indications: Vec<String>,
    /// Device specifications
    pub specifications: HashMap<String, String>,
    /// Key features
    pub features: Vec<String>,
    /// Contraindications
    pub contraindications: Vec<String>,
}

impl DeviceDescription {
    /// Create a new device description
    pub fn new(name: impl Into<String>, classification: DeviceClass) -> Self {
        Self {
            name: name.into(),
            classification,
            intended_use: String::new(),
            indications: Vec::new(),
            specifications: HashMap::new(),
            features: Vec::new(),
            contraindications: Vec::new(),
        }
    }

    /// Set intended use statement
    pub fn with_intended_use(mut self, use_statement: impl Into<String>) -> Self {
        self.intended_use = use_statement.into();
        self
    }

    /// Add indication for use
    pub fn add_indication(&mut self, indication: impl Into<String>) {
        self.indications.push(indication.into());
    }

    /// Add device specification
    pub fn add_specification(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.specifications.insert(key.into(), value.into());
    }

    /// Add feature
    pub fn add_feature(&mut self, feature: impl Into<String>) {
        self.features.push(feature.into());
    }

    /// Add contraindication
    pub fn add_contraindication(&mut self, contraindication: impl Into<String>) {
        self.contraindications.push(contraindication.into());
    }

    /// Validate device description completeness
    pub fn validate(&self) -> KwaversResult<()> {
        if self.name.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Device name cannot be empty".to_string(),
            ));
        }

        if self.intended_use.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Intended use statement required".to_string(),
            ));
        }

        if self.indications.is_empty() {
            return Err(KwaversError::InvalidInput(
                "At least one indication for use required".to_string(),
            ));
        }

        Ok(())
    }
}

/// Predicate device information for substantial equivalence
#[derive(Debug, Clone)]
pub struct PredicateDevice {
    /// Predicate device name
    pub name: String,
    /// 510(k) number of predicate
    pub k_number: String,
    /// Manufacturer name
    pub manufacturer: String,
    /// Year of predicate clearance
    pub clearance_year: u32,
    /// Similarities to predicate
    pub similarities: Vec<String>,
    /// Differences from predicate (and justification)
    pub differences: Vec<(String, String)>,
}

impl PredicateDevice {
    /// Create new predicate device record
    pub fn new(
        name: impl Into<String>,
        k_number: impl Into<String>,
        manufacturer: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            k_number: k_number.into(),
            manufacturer: manufacturer.into(),
            clearance_year: 2020,
            similarities: Vec::new(),
            differences: Vec::new(),
        }
    }

    /// Add similarity to predicate
    pub fn add_similarity(&mut self, similarity: impl Into<String>) {
        self.similarities.push(similarity.into());
    }

    /// Add difference with justification
    pub fn add_difference(
        &mut self,
        difference: impl Into<String>,
        justification: impl Into<String>,
    ) {
        self.differences
            .push((difference.into(), justification.into()));
    }

    /// Validate predicate device information
    pub fn validate(&self) -> KwaversResult<()> {
        if self.name.is_empty() || self.k_number.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Predicate device name and K number required".to_string(),
            ));
        }

        if self.similarities.is_empty() {
            return Err(KwaversError::InvalidInput(
                "At least one similarity to predicate required".to_string(),
            ));
        }

        Ok(())
    }
}

/// Risk management record
#[derive(Debug, Clone)]
pub struct RiskRecord {
    /// Risk ID
    pub risk_id: String,
    /// Risk description
    pub description: String,
    /// Severity level: Low, Medium, High, Critical
    pub severity: String,
    /// Probability of occurrence: Low, Medium, High
    pub probability: String,
    /// Risk priority number (RPN)
    pub rpn: u32,
    /// Mitigation measures
    pub mitigations: Vec<String>,
    /// Residual risk assessment
    pub residual_assessment: String,
}

impl RiskRecord {
    /// Create new risk record
    pub fn new(risk_id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            risk_id: risk_id.into(),
            description: description.into(),
            severity: "Medium".to_string(),
            probability: "Medium".to_string(),
            rpn: 9,
            mitigations: Vec::new(),
            residual_assessment: String::new(),
        }
    }

    /// Set severity level
    pub fn with_severity(mut self, severity: impl Into<String>) -> Self {
        self.severity = severity.into();
        self
    }

    /// Set probability level
    pub fn with_probability(mut self, probability: impl Into<String>) -> Self {
        self.probability = probability.into();
        self
    }

    /// Calculate RPN based on severity and probability
    pub fn calculate_rpn(&mut self) -> u32 {
        let severity = match self.severity.as_str() {
            "Critical" => 10,
            "High" => 7,
            "Medium" => 5,
            "Low" => 2,
            _ => 3,
        };

        let probability = match self.probability.as_str() {
            "High" => 5,
            "Medium" => 3,
            "Low" => 1,
            _ => 2,
        };

        self.rpn = severity * probability;
        self.rpn
    }

    /// Add mitigation measure
    pub fn add_mitigation(&mut self, mitigation: impl Into<String>) {
        self.mitigations.push(mitigation.into());
    }

    /// Set residual risk assessment
    pub fn set_residual_assessment(&mut self, assessment: impl Into<String>) {
        self.residual_assessment = assessment.into();
    }
}

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

/// Clinical evidence summary
#[derive(Debug, Clone)]
pub struct ClinicalEvidence {
    /// Evidence reference ID
    pub ref_id: String,
    /// Study title
    pub title: String,
    /// Study type: Case Study, Case Series, Clinical Trial, etc.
    pub study_type: String,
    /// Number of subjects
    pub subject_count: u32,
    /// Study duration (days)
    pub duration_days: u32,
    /// Primary outcome
    pub primary_outcome: String,
    /// Key findings
    pub key_findings: Vec<String>,
    /// Adverse events reported
    pub adverse_events: Vec<String>,
}

impl ClinicalEvidence {
    /// Create new clinical evidence record
    pub fn new(ref_id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            ref_id: ref_id.into(),
            title: title.into(),
            study_type: "Clinical Trial".to_string(),
            subject_count: 0,
            duration_days: 0,
            primary_outcome: String::new(),
            key_findings: Vec::new(),
            adverse_events: Vec::new(),
        }
    }

    /// Set study type
    pub fn with_study_type(mut self, study_type: impl Into<String>) -> Self {
        self.study_type = study_type.into();
        self
    }

    /// Add key finding
    pub fn add_finding(&mut self, finding: impl Into<String>) {
        self.key_findings.push(finding.into());
    }

    /// Add adverse event
    pub fn add_adverse_event(&mut self, event: impl Into<String>) {
        self.adverse_events.push(event.into());
    }

    /// Validate clinical evidence
    pub fn validate(&self) -> KwaversResult<()> {
        if self.subject_count == 0 {
            return Err(KwaversError::InvalidInput(
                "Subject count must be greater than zero".to_string(),
            ));
        }

        if self.primary_outcome.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Primary outcome must be specified".to_string(),
            ));
        }

        Ok(())
    }
}

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

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current ISO 8601 datetime string
fn iso8601_now() -> String {
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();

    let seconds = duration.as_secs();
    let days_since_epoch = seconds / 86400;
    let year = 1970 + days_since_epoch / 365;
    let month = (days_since_epoch % 365) / 30 + 1;
    let day = (days_since_epoch % 365) % 30 + 1;

    format!("{:04}-{:02}-{:02}T00:00:00Z", year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_class() {
        assert_eq!(DeviceClass::ClassII.as_str(), "Class II");
        assert_eq!(DeviceClass::ClassII.risk_level(), "Moderate Risk");
    }

    #[test]
    fn test_device_description_creation() {
        let desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        assert_eq!(desc.name, "HIFU Device");
        assert_eq!(desc.classification, DeviceClass::ClassII);
    }

    #[test]
    fn test_device_description_validation() {
        let desc = DeviceDescription::new("", DeviceClass::ClassII);
        assert!(desc.validate().is_err());

        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        assert!(desc.validate().is_err()); // No intended use

        desc.intended_use = "Ablation therapy".to_string();
        assert!(desc.validate().is_err()); // No indications

        desc.add_indication("Benign tumors");
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_predicate_device() {
        let predicate = PredicateDevice::new("Device A", "K060123", "Manufacturer");
        assert_eq!(predicate.k_number, "K060123");
    }

    #[test]
    fn test_risk_record_rpn_calculation() {
        let mut risk = RiskRecord::new("R001", "Device failure");
        risk.severity = "High".to_string();
        risk.probability = "Medium".to_string();

        let rpn = risk.calculate_rpn();
        assert_eq!(rpn, 7 * 3); // 21
    }

    #[test]
    fn test_performance_test() {
        let test =
            PerformanceTest::new("TEST001", "Safety Test").with_objective("Verify device safety");
        assert_eq!(test.objective, "Verify device safety");
        assert!(test.result.is_none());
    }

    #[test]
    fn test_clinical_evidence() {
        let mut evidence = ClinicalEvidence::new("EV001", "Clinical Study");
        evidence.subject_count = 50;
        evidence.primary_outcome = "Pain reduction".to_string();

        assert!(evidence.validate().is_ok());
    }

    #[test]
    fn test_submission_document_creation() {
        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        desc.intended_use = "Ablation therapy".to_string();
        desc.add_indication("Benign tumors");

        let doc = SubmissionDocument::new("Manufacturer Inc", desc);
        assert!(doc.is_ok());

        let doc = doc.unwrap();
        assert_eq!(doc.status, "Draft");
    }

    #[test]
    fn test_submission_checklist() {
        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        desc.intended_use = "Ablation therapy".to_string();
        desc.add_indication("Benign tumors");

        let doc = SubmissionDocument::new("Manufacturer Inc", desc).unwrap();
        let checklist = doc.generate_checklist();

        assert!(checklist.contains_key("Device Description Complete"));
        assert!(*checklist.get("Device Description Complete").unwrap());
    }

    #[test]
    fn test_submission_validation() {
        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        desc.intended_use = "Ablation therapy".to_string();
        desc.add_indication("Benign tumors");

        let mut doc = SubmissionDocument::new("Manufacturer Inc", desc).unwrap();

        // Should fail - no predicate device
        assert!(doc.validate().is_err());

        // Add predicate
        let predicate = PredicateDevice::new("Device A", "K060123", "Manufacturer");
        let mut predicate_fixed = predicate.clone();
        predicate_fixed.add_similarity("Same intended use");
        doc.add_predicate(predicate_fixed).ok();

        // Should fail - no performance tests
        assert!(doc.validate().is_err());

        // Add test
        let mut test = PerformanceTest::new("TEST001", "Safety Test");
        test.objective = "Verify safety".to_string();
        test.method = "ISO standard".to_string();
        test.record_result(true);
        doc.add_performance_test(test).ok();

        // Should fail - no summary
        assert!(doc.validate().is_err());

        doc.summary = "This device is equivalent to predicate".to_string();

        // Should pass now
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn test_submission_statistics() {
        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        desc.intended_use = "Ablation therapy".to_string();
        desc.add_indication("Benign tumors");

        let doc = SubmissionDocument::new("Manufacturer Inc", desc).unwrap();
        let stats = doc.statistics();

        assert!(stats.contains_key("Device Name"));
        assert_eq!(stats.get("Device Name").unwrap(), "HIFU Device");
    }

    #[test]
    fn test_submission_submit() {
        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        desc.intended_use = "Ablation therapy".to_string();
        desc.add_indication("Benign tumors");

        let mut doc = SubmissionDocument::new("Manufacturer Inc", desc).unwrap();

        // Setup minimum requirements
        let mut predicate = PredicateDevice::new("Device A", "K060123", "Manufacturer");
        predicate.add_similarity("Same intended use");
        doc.add_predicate(predicate).ok();

        let mut test = PerformanceTest::new("TEST001", "Safety Test");
        test.objective = "Verify safety".to_string();
        test.method = "ISO standard".to_string();
        test.record_result(true);
        doc.add_performance_test(test).ok();

        doc.summary = "This device is equivalent to predicate".to_string();

        // Should succeed
        assert!(doc.submit().is_ok());
        assert_eq!(doc.status, "Submitted");
    }
}
