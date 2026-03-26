use super::demographics::PatientDemographics;
use crate::core::error::KwaversResult;
use std::time::{SystemTime, UNIX_EPOCH};

/// Get current Unix timestamp in milliseconds
pub(crate) fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Get current ISO 8601 datetime string
pub(crate) fn iso8601_now() -> String {
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();

    let seconds = duration.as_secs();
    let days_since_epoch = seconds / 86400;
    let year = 1970 + days_since_epoch / 365;
    let month = (days_since_epoch % 365) / 30 + 1;
    let day = (days_since_epoch % 365) % 30 + 1;

    format!("{:04}-{:02}-{:02}T00:00:00Z", year, month, day)
}

/// Medical history entry
#[derive(Debug, Clone)]
pub struct MedicalHistoryEntry {
    /// Diagnosis code (ICD-10)
    pub diagnosis_code: String,
    /// Diagnosis description
    pub diagnosis: String,
    /// Date of diagnosis (ISO 8601)
    pub date: String,
    /// Status: active, resolved, rule-out
    pub status: String,
    /// Clinical notes
    pub notes: String,
}

/// Medication record
#[derive(Debug, Clone)]
pub struct MedicationRecord {
    /// Medication name
    pub medication: String,
    /// Dosage amount and unit (e.g., "500mg")
    pub dosage: String,
    /// Frequency (e.g., "twice daily")
    pub frequency: String,
    /// Start date (ISO 8601)
    pub start_date: String,
    /// End date if discontinued
    pub end_date: Option<String>,
    /// Indication/reason for medication
    pub indication: String,
}

/// Patient medical profile
#[derive(Debug, Clone)]
pub struct PatientMedicalProfile {
    /// Patient demographics
    pub demographics: PatientDemographics,
    /// Active and historical diagnoses
    pub medical_history: Vec<MedicalHistoryEntry>,
    /// Current medications
    pub medications: Vec<MedicationRecord>,
    /// Allergies (format: "Allergen - Reaction")
    pub allergies: Vec<String>,
    /// Surgical history
    pub surgical_history: Vec<String>,
    /// Last updated timestamp (Unix seconds)
    pub last_updated: u64,
}

impl PatientMedicalProfile {
    /// Create a new patient profile
    pub fn new(demographics: PatientDemographics) -> KwaversResult<Self> {
        demographics.validate()?;

        Ok(Self {
            demographics,
            medical_history: Vec::new(),
            medications: Vec::new(),
            allergies: Vec::new(),
            surgical_history: Vec::new(),
            last_updated: current_timestamp(),
        })
    }

    /// Add a diagnosis to medical history
    pub fn add_diagnosis(
        &mut self,
        diagnosis_code: impl Into<String>,
        diagnosis: impl Into<String>,
        notes: impl Into<String>,
    ) -> KwaversResult<()> {
        let entry = MedicalHistoryEntry {
            diagnosis_code: diagnosis_code.into(),
            diagnosis: diagnosis.into(),
            date: iso8601_now(),
            status: "active".to_string(),
            notes: notes.into(),
        };

        self.medical_history.push(entry);
        self.last_updated = current_timestamp();
        Ok(())
    }

    /// Add a medication
    pub fn add_medication(
        &mut self,
        medication: impl Into<String>,
        dosage: impl Into<String>,
        frequency: impl Into<String>,
        indication: impl Into<String>,
    ) -> KwaversResult<()> {
        let record = MedicationRecord {
            medication: medication.into(),
            dosage: dosage.into(),
            frequency: frequency.into(),
            start_date: iso8601_now(),
            end_date: None,
            indication: indication.into(),
        };

        self.medications.push(record);
        self.last_updated = current_timestamp();
        Ok(())
    }

    /// Add an allergy
    pub fn add_allergy(&mut self, allergen: impl Into<String>, reaction: impl Into<String>) {
        let allergen_str = allergen.into();
        let reaction_str = reaction.into();
        self.allergies
            .push(format!("{} - {}", allergen_str, reaction_str));
        self.last_updated = current_timestamp();
    }
}
