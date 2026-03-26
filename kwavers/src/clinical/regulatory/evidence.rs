use crate::core::error::{KwaversError, KwaversResult};

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
