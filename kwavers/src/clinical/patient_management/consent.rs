use super::demographics::PatientId;
use super::profile::iso8601_now;

/// Consent type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentType {
    /// General treatment consent
    GeneralTreatment,
    /// Imaging consent
    Imaging,
    /// Procedure-specific consent
    Procedure,
    /// Research participation
    Research,
    /// HIPAA authorization
    HipaaAuthorization,
    /// Data sharing consent
    DataSharing,
}

impl ConsentType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GeneralTreatment => "General Treatment",
            Self::Imaging => "Imaging",
            Self::Procedure => "Procedure",
            Self::Research => "Research",
            Self::HipaaAuthorization => "HIPAA Authorization",
            Self::DataSharing => "Data Sharing",
        }
    }
}

/// Consent record
#[derive(Debug, Clone)]
pub struct ConsentRecord {
    /// Consent ID
    pub consent_id: String,
    /// Patient ID
    pub patient_id: PatientId,
    /// Type of consent
    pub consent_type: ConsentType,
    /// Consent date (ISO 8601)
    pub date: String,
    /// Expiration date if applicable
    pub expiration_date: Option<String>,
    /// Clinician who obtained consent
    pub clinician: String,
    /// Status: active, expired, revoked
    pub status: String,
    /// Consent document reference/hash
    pub document_reference: String,
}

impl ConsentRecord {
    /// Create a new consent record
    pub fn new(
        patient_id: PatientId,
        consent_type: ConsentType,
        clinician: impl Into<String>,
    ) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);

        Self {
            consent_id: format!("CONSENT_{:08}", id),
            patient_id,
            consent_type,
            date: iso8601_now(),
            expiration_date: None,
            clinician: clinician.into(),
            status: "active".to_string(),
            document_reference: format!("consent_{:08}", id),
        }
    }

    /// Check if consent is currently valid
    pub fn is_valid(&self) -> bool {
        if self.status != "active" {
            return false;
        }

        if let Some(expiry) = &self.expiration_date {
            // Simple ISO 8601 string comparison (works for dates)
            expiry > &iso8601_now()
        } else {
            true
        }
    }

    /// Revoke consent
    pub fn revoke(&mut self) {
        self.status = "revoked".to_string();
    }
}
