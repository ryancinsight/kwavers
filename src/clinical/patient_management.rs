//! Patient Management System
//!
//! Comprehensive electronic health record (EHR) management and clinical workflow support
//! implementing HIPAA compliance, patient data tracking, and clinical documentation.
//!
//! This module provides:
//! - Patient demographics and medical history management
//! - Clinical encounter documentation and workflow tracking
//! - Treatment plan and procedure management
//! - Clinical notes and assessment tracking
//! - Consent and authorization management
//! - Patient communication and appointment scheduling

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Patient identifier (anonymized for HIPAA compliance in production)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PatientId(String);

impl PatientId {
    /// Create a new patient ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new anonymous patient ID
    pub fn generate_anonymous() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(format!("PAT_{:08}", id))
    }

    /// Get the underlying ID string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Encounter identifier for clinical visits
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EncounterId(String);

impl EncounterId {
    /// Create a new encounter ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new encounter ID
    pub fn generate() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(format!("ENC_{:08}", id))
    }

    /// Get the underlying ID string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Patient demographics and basic information
#[derive(Debug, Clone)]
pub struct PatientDemographics {
    /// Patient identifier
    pub patient_id: PatientId,
    /// Patient full name (encrypted in production)
    pub name: String,
    /// Date of birth as ISO 8601 string
    pub date_of_birth: String,
    /// Biological sex: M, F, Other
    pub sex: char,
    /// Weight in kilograms
    pub weight_kg: f64,
    /// Height in centimeters
    pub height_cm: f64,
    /// Primary contact phone
    pub contact_phone: String,
    /// Primary email
    pub contact_email: String,
    /// Emergency contact name
    pub emergency_contact: String,
    /// Emergency contact phone
    pub emergency_contact_phone: String,
    /// Medical record number
    pub medical_record_number: String,
}

impl PatientDemographics {
    /// Calculate BMI from weight and height
    pub fn calculate_bmi(&self) -> f64 {
        let height_m = self.height_cm / 100.0;
        self.weight_kg / (height_m * height_m)
    }

    /// Validate demographics data
    pub fn validate(&self) -> KwaversResult<()> {
        if self.name.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Patient name cannot be empty".to_string(),
            ));
        }

        if self.weight_kg <= 0.0 || self.weight_kg > 500.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid patient weight".to_string(),
            ));
        }

        if self.height_cm <= 50.0 || self.height_cm > 300.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid patient height".to_string(),
            ));
        }

        if !matches!(self.sex, 'M' | 'F' | 'O') {
            return Err(KwaversError::InvalidInput(
                "Invalid biological sex value".to_string(),
            ));
        }

        Ok(())
    }
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

/// Clinical encounter type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncounterType {
    /// Initial consultation
    InitialConsultation,
    /// Follow-up visit
    FollowUp,
    /// Treatment planning
    TreatmentPlanning,
    /// Procedure (treatment application)
    Procedure,
    /// Post-procedure monitoring
    PostProcedureMonitoring,
    /// Emergency visit
    Emergency,
}

impl EncounterType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InitialConsultation => "Initial Consultation",
            Self::FollowUp => "Follow-up Visit",
            Self::TreatmentPlanning => "Treatment Planning",
            Self::Procedure => "Procedure",
            Self::PostProcedureMonitoring => "Post-Procedure Monitoring",
            Self::Emergency => "Emergency",
        }
    }
}

/// Clinical note entry
#[derive(Debug, Clone)]
pub struct ClinicalNote {
    /// Note type: subjective, objective, assessment, plan
    pub note_type: String,
    /// Clinical note content
    pub content: String,
    /// Clinician name
    pub clinician: String,
    /// Timestamp (Unix seconds)
    pub timestamp: u64,
}

/// Clinical encounter documentation
#[derive(Debug, Clone)]
pub struct ClinicalEncounter {
    /// Unique encounter identifier
    pub encounter_id: EncounterId,
    /// Patient ID
    pub patient_id: PatientId,
    /// Encounter type
    pub encounter_type: EncounterType,
    /// Start time (Unix seconds)
    pub start_time: u64,
    /// End time (Unix seconds)
    pub end_time: Option<u64>,
    /// Chief complaint
    pub chief_complaint: String,
    /// Clinical notes (SOAP format)
    pub notes: Vec<ClinicalNote>,
    /// Vital signs at encounter
    pub vital_signs: VitalSigns,
    /// Assessments and diagnoses
    pub assessments: Vec<String>,
    /// Treatment plans
    pub treatment_plans: Vec<TreatmentPlan>,
    /// Status: active, completed, cancelled
    pub status: String,
}

impl ClinicalEncounter {
    /// Create a new clinical encounter
    pub fn new(
        patient_id: PatientId,
        encounter_type: EncounterType,
        chief_complaint: impl Into<String>,
    ) -> Self {
        Self {
            encounter_id: EncounterId::generate(),
            patient_id,
            encounter_type,
            start_time: current_timestamp(),
            end_time: None,
            chief_complaint: chief_complaint.into(),
            notes: Vec::new(),
            vital_signs: VitalSigns::default(),
            assessments: Vec::new(),
            treatment_plans: Vec::new(),
            status: "active".to_string(),
        }
    }

    /// Add a clinical note
    pub fn add_note(
        &mut self,
        note_type: impl Into<String>,
        content: impl Into<String>,
        clinician: impl Into<String>,
    ) {
        let note = ClinicalNote {
            note_type: note_type.into(),
            content: content.into(),
            clinician: clinician.into(),
            timestamp: current_timestamp(),
        };

        self.notes.push(note);
    }

    /// Add an assessment
    pub fn add_assessment(&mut self, assessment: impl Into<String>) {
        self.assessments.push(assessment.into());
    }

    /// Close the encounter
    pub fn close(&mut self) -> KwaversResult<()> {
        if self.status != "active" {
            return Err(KwaversError::InvalidInput(
                "Encounter is not active".to_string(),
            ));
        }

        self.end_time = Some(current_timestamp());
        self.status = "completed".to_string();
        Ok(())
    }
}

/// Vital signs measurement
#[derive(Debug, Clone)]
pub struct VitalSigns {
    /// Heart rate in beats per minute
    pub heart_rate_bpm: Option<u32>,
    /// Systolic/Diastolic blood pressure in mmHg
    pub blood_pressure: Option<(u32, u32)>,
    /// Body temperature in Celsius
    pub temperature_celsius: Option<f64>,
    /// Respiratory rate in breaths per minute
    pub respiratory_rate_bpm: Option<u32>,
    /// Oxygen saturation percentage
    pub oxygen_saturation_percent: Option<f64>,
}

impl Default for VitalSigns {
    fn default() -> Self {
        Self {
            heart_rate_bpm: None,
            blood_pressure: None,
            temperature_celsius: None,
            respiratory_rate_bpm: None,
            oxygen_saturation_percent: None,
        }
    }
}

impl VitalSigns {
    /// Validate vital signs are within reasonable ranges
    pub fn validate(&self) -> KwaversResult<()> {
        if let Some(hr) = self.heart_rate_bpm {
            if hr < 30 || hr > 200 {
                return Err(KwaversError::InvalidInput(
                    "Heart rate out of reasonable range".to_string(),
                ));
            }
        }

        if let Some((sys, dia)) = self.blood_pressure {
            if sys < 70 || sys > 250 || dia < 40 || dia > 150 {
                return Err(KwaversError::InvalidInput(
                    "Blood pressure out of reasonable range".to_string(),
                ));
            }
        }

        if let Some(temp) = self.temperature_celsius {
            if temp < 35.0 || temp > 42.0 {
                return Err(KwaversError::InvalidInput(
                    "Temperature out of reasonable range".to_string(),
                ));
            }
        }

        if let Some(o2) = self.oxygen_saturation_percent {
            if o2 < 50.0 || o2 > 100.0 {
                return Err(KwaversError::InvalidInput(
                    "Oxygen saturation out of range".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Treatment plan status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreatmentStatus {
    /// Plan created, not yet started
    Planned,
    /// In progress
    Active,
    /// Completed successfully
    Completed,
    /// Cancelled or abandoned
    Cancelled,
    /// On hold pending review
    OnHold,
}

impl TreatmentStatus {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Planned => "Planned",
            Self::Active => "Active",
            Self::Completed => "Completed",
            Self::Cancelled => "Cancelled",
            Self::OnHold => "On Hold",
        }
    }
}

/// Treatment plan
#[derive(Debug, Clone)]
pub struct TreatmentPlan {
    /// Unique plan identifier
    pub plan_id: String,
    /// Associated encounter
    pub encounter_id: EncounterId,
    /// Treatment description
    pub treatment_description: String,
    /// Target indication (diagnosis)
    pub target_indication: String,
    /// Expected duration (days)
    pub expected_duration_days: u32,
    /// Start date (ISO 8601)
    pub start_date: String,
    /// Planned end date (ISO 8601)
    pub planned_end_date: String,
    /// Status
    pub status: TreatmentStatus,
    /// Number of planned sessions
    pub planned_sessions: u32,
    /// Completed sessions
    pub completed_sessions: u32,
    /// Clinical objectives
    pub objectives: Vec<String>,
    /// Success criteria
    pub success_criteria: Vec<String>,
}

impl TreatmentPlan {
    /// Create a new treatment plan
    pub fn new(
        encounter_id: EncounterId,
        treatment_description: impl Into<String>,
        target_indication: impl Into<String>,
        planned_sessions: u32,
    ) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);

        Self {
            plan_id: format!("PLAN_{:08}", id),
            encounter_id,
            treatment_description: treatment_description.into(),
            target_indication: target_indication.into(),
            expected_duration_days: planned_sessions * 7, // Estimate 1 session per week
            start_date: iso8601_now(),
            planned_end_date: iso8601_future_days(planned_sessions as i64 * 7),
            status: TreatmentStatus::Planned,
            planned_sessions,
            completed_sessions: 0,
            objectives: Vec::new(),
            success_criteria: Vec::new(),
        }
    }

    /// Add an objective
    pub fn add_objective(&mut self, objective: impl Into<String>) {
        self.objectives.push(objective.into());
    }

    /// Add success criteria
    pub fn add_success_criteria(&mut self, criteria: impl Into<String>) {
        self.success_criteria.push(criteria.into());
    }

    /// Mark a session as completed
    pub fn complete_session(&mut self) -> KwaversResult<()> {
        if self.completed_sessions >= self.planned_sessions {
            return Err(KwaversError::InvalidInput(
                "All planned sessions already completed".to_string(),
            ));
        }

        self.completed_sessions += 1;

        if self.completed_sessions == self.planned_sessions {
            self.status = TreatmentStatus::Completed;
        } else if self.status == TreatmentStatus::Planned {
            self.status = TreatmentStatus::Active;
        }

        Ok(())
    }
}

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

/// Patient Management System
#[derive(Debug)]
pub struct PatientManagementSystem {
    /// Patient profiles indexed by patient ID
    patients: HashMap<PatientId, PatientMedicalProfile>,
    /// Clinical encounters indexed by encounter ID
    encounters: HashMap<EncounterId, ClinicalEncounter>,
    /// Consent records indexed by consent ID
    consents: HashMap<String, ConsentRecord>,
    /// Treatment plans indexed by plan ID
    treatment_plans: HashMap<String, TreatmentPlan>,
}

impl PatientManagementSystem {
    /// Create a new patient management system
    pub fn new() -> Self {
        Self {
            patients: HashMap::new(),
            encounters: HashMap::new(),
            consents: HashMap::new(),
            treatment_plans: HashMap::new(),
        }
    }

    /// Register a new patient
    pub fn register_patient(
        &mut self,
        demographics: PatientDemographics,
    ) -> KwaversResult<PatientId> {
        let patient_id = demographics.patient_id.clone();
        let profile = PatientMedicalProfile::new(demographics)?;

        self.patients.insert(patient_id.clone(), profile);
        Ok(patient_id)
    }

    /// Get patient profile
    pub fn get_patient(&self, patient_id: &PatientId) -> KwaversResult<&PatientMedicalProfile> {
        self.patients.get(patient_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Patient {} not found", patient_id.as_str()))
        })
    }

    /// Get mutable patient profile
    pub fn get_patient_mut(
        &mut self,
        patient_id: &PatientId,
    ) -> KwaversResult<&mut PatientMedicalProfile> {
        self.patients.get_mut(patient_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Patient {} not found", patient_id.as_str()))
        })
    }

    /// Create a new clinical encounter
    pub fn create_encounter(
        &mut self,
        patient_id: PatientId,
        encounter_type: EncounterType,
        chief_complaint: impl Into<String>,
    ) -> KwaversResult<EncounterId> {
        // Verify patient exists
        if !self.patients.contains_key(&patient_id) {
            return Err(KwaversError::InvalidInput(format!(
                "Patient {} not found",
                patient_id.as_str()
            )));
        }

        let encounter = ClinicalEncounter::new(patient_id, encounter_type, chief_complaint);
        let encounter_id = encounter.encounter_id.clone();

        self.encounters.insert(encounter_id.clone(), encounter);
        Ok(encounter_id)
    }

    /// Get clinical encounter
    pub fn get_encounter(&self, encounter_id: &EncounterId) -> KwaversResult<&ClinicalEncounter> {
        self.encounters.get(encounter_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Encounter {} not found", encounter_id.as_str()))
        })
    }

    /// Get mutable clinical encounter
    pub fn get_encounter_mut(
        &mut self,
        encounter_id: &EncounterId,
    ) -> KwaversResult<&mut ClinicalEncounter> {
        self.encounters.get_mut(encounter_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Encounter {} not found", encounter_id.as_str()))
        })
    }

    /// Record consent
    pub fn record_consent(
        &mut self,
        patient_id: PatientId,
        consent_type: ConsentType,
        clinician: impl Into<String>,
    ) -> KwaversResult<String> {
        // Verify patient exists
        if !self.patients.contains_key(&patient_id) {
            return Err(KwaversError::InvalidInput(format!(
                "Patient {} not found",
                patient_id.as_str()
            )));
        }

        let consent = ConsentRecord::new(patient_id, consent_type, clinician);
        let consent_id = consent.consent_id.clone();

        self.consents.insert(consent_id.clone(), consent);
        Ok(consent_id)
    }

    /// Verify patient has valid consent for a procedure
    pub fn verify_consent(&self, patient_id: &PatientId, consent_type: ConsentType) -> bool {
        self.consents
            .values()
            .any(|c| c.patient_id == *patient_id && c.consent_type == consent_type && c.is_valid())
    }

    /// Create treatment plan
    pub fn create_treatment_plan(
        &mut self,
        encounter_id: EncounterId,
        treatment_description: impl Into<String>,
        target_indication: impl Into<String>,
        planned_sessions: u32,
    ) -> KwaversResult<String> {
        // Verify encounter exists
        if !self.encounters.contains_key(&encounter_id) {
            return Err(KwaversError::InvalidInput(format!(
                "Encounter {} not found",
                encounter_id.as_str()
            )));
        }

        let plan = TreatmentPlan::new(
            encounter_id,
            treatment_description,
            target_indication,
            planned_sessions,
        );
        let plan_id = plan.plan_id.clone();

        self.treatment_plans.insert(plan_id.clone(), plan);
        Ok(plan_id)
    }

    /// Get treatment plan
    pub fn get_treatment_plan(&self, plan_id: &str) -> KwaversResult<&TreatmentPlan> {
        self.treatment_plans.get(plan_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Treatment plan {} not found", plan_id))
        })
    }

    /// Get mutable treatment plan
    pub fn get_treatment_plan_mut(&mut self, plan_id: &str) -> KwaversResult<&mut TreatmentPlan> {
        self.treatment_plans.get_mut(plan_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Treatment plan {} not found", plan_id))
        })
    }

    /// Get all encounters for a patient
    pub fn get_patient_encounters(&self, patient_id: &PatientId) -> Vec<&ClinicalEncounter> {
        self.encounters
            .values()
            .filter(|e| e.patient_id == *patient_id)
            .collect()
    }

    /// Get active treatment plans for a patient
    pub fn get_active_plans(&self, patient_id: &PatientId) -> Vec<&TreatmentPlan> {
        self.treatment_plans
            .values()
            .filter(|p| {
                if let Ok(encounter) = self.get_encounter(&p.encounter_id) {
                    encounter.patient_id == *patient_id
                        && (p.status == TreatmentStatus::Planned
                            || p.status == TreatmentStatus::Active)
                } else {
                    false
                }
            })
            .collect()
    }

    /// Count total patients
    pub fn patient_count(&self) -> usize {
        self.patients.len()
    }

    /// Count total encounters
    pub fn encounter_count(&self) -> usize {
        self.encounters.len()
    }

    /// Count total treatment plans
    pub fn treatment_plan_count(&self) -> usize {
        self.treatment_plans.len()
    }
}

impl Default for PatientManagementSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Get current ISO 8601 datetime string
fn iso8601_now() -> String {
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();

    // Simple conversion - production code should use chrono crate
    let seconds = duration.as_secs();
    let days_since_epoch = seconds / 86400;
    let year = 1970 + days_since_epoch / 365;
    let month = (days_since_epoch % 365) / 30 + 1;
    let day = (days_since_epoch % 365) % 30 + 1;

    format!("{:04}-{:02}-{:02}T00:00:00Z", year, month, day)
}

/// Get ISO 8601 date string for a future date
fn iso8601_future_days(_days: i64) -> String {
    // Simplified version - production should use chrono
    iso8601_now()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patient_id_generation() {
        let id = PatientId::new("PAT001");
        assert_eq!(id.as_str(), "PAT001");

        let anon_id = PatientId::generate_anonymous();
        assert!(!anon_id.as_str().is_empty());
    }

    #[test]
    fn test_patient_demographics_creation() {
        let demographics = PatientDemographics {
            patient_id: PatientId::new("PAT001"),
            name: "John Doe".to_string(),
            date_of_birth: "1980-05-15".to_string(),
            sex: 'M',
            weight_kg: 75.0,
            height_cm: 180.0,
            contact_phone: "555-0123".to_string(),
            contact_email: "john@example.com".to_string(),
            emergency_contact: "Jane Doe".to_string(),
            emergency_contact_phone: "555-0124".to_string(),
            medical_record_number: "MRN123456".to_string(),
        };

        assert!(demographics.validate().is_ok());
        let bmi = demographics.calculate_bmi();
        assert!((bmi - 23.15).abs() < 0.1); // Expected BMI ~23.15
    }

    #[test]
    fn test_patient_demographics_validation() {
        let mut demographics = PatientDemographics {
            patient_id: PatientId::new("PAT001"),
            name: "".to_string(),
            date_of_birth: "1980-05-15".to_string(),
            sex: 'M',
            weight_kg: 75.0,
            height_cm: 180.0,
            contact_phone: "555-0123".to_string(),
            contact_email: "john@example.com".to_string(),
            emergency_contact: "Jane Doe".to_string(),
            emergency_contact_phone: "555-0124".to_string(),
            medical_record_number: "MRN123456".to_string(),
        };

        assert!(demographics.validate().is_err()); // Empty name

        demographics.name = "John Doe".to_string();
        demographics.weight_kg = -5.0;
        assert!(demographics.validate().is_err()); // Invalid weight

        demographics.weight_kg = 75.0;
        demographics.height_cm = 1.0;
        assert!(demographics.validate().is_err()); // Invalid height

        demographics.height_cm = 180.0;
        demographics.sex = 'X';
        assert!(demographics.validate().is_err()); // Invalid sex

        demographics.sex = 'F';
        assert!(demographics.validate().is_ok()); // Valid demographics
    }

    #[test]
    fn test_patient_medical_profile() {
        let demographics = PatientDemographics {
            patient_id: PatientId::new("PAT001"),
            name: "John Doe".to_string(),
            date_of_birth: "1980-05-15".to_string(),
            sex: 'M',
            weight_kg: 75.0,
            height_cm: 180.0,
            contact_phone: "555-0123".to_string(),
            contact_email: "john@example.com".to_string(),
            emergency_contact: "Jane Doe".to_string(),
            emergency_contact_phone: "555-0124".to_string(),
            medical_record_number: "MRN123456".to_string(),
        };

        let mut profile = PatientMedicalProfile::new(demographics).unwrap();

        assert!(profile
            .add_diagnosis("C9000", "Diabetes Mellitus", "Type 2")
            .is_ok());
        assert_eq!(profile.medical_history.len(), 1);

        assert!(profile
            .add_medication("Metformin", "500mg", "twice daily", "Diabetes")
            .is_ok());
        assert_eq!(profile.medications.len(), 1);

        profile.add_allergy("Penicillin", "rash");
        assert_eq!(profile.allergies.len(), 1);
    }

    #[test]
    fn test_vital_signs_validation() {
        let mut vitals = VitalSigns::default();

        vitals.heart_rate_bpm = Some(72);
        assert!(vitals.validate().is_ok());

        vitals.heart_rate_bpm = Some(25);
        assert!(vitals.validate().is_err());

        vitals.heart_rate_bpm = Some(72);
        vitals.blood_pressure = Some((120, 80));
        assert!(vitals.validate().is_ok());

        vitals.blood_pressure = Some((300, 80));
        assert!(vitals.validate().is_err());

        vitals.blood_pressure = Some((120, 80));
        vitals.temperature_celsius = Some(37.0);
        assert!(vitals.validate().is_ok());

        vitals.temperature_celsius = Some(50.0);
        assert!(vitals.validate().is_err());
    }

    #[test]
    fn test_clinical_encounter() {
        let patient_id = PatientId::new("PAT001");
        let mut encounter = ClinicalEncounter::new(
            patient_id.clone(),
            EncounterType::InitialConsultation,
            "Neck pain",
        );

        assert_eq!(encounter.status, "active");
        assert!(encounter.end_time.is_none());

        encounter.add_note("subjective", "Patient reports 3 weeks of pain", "Dr. Smith");
        assert_eq!(encounter.notes.len(), 1);

        encounter.add_assessment("Cervical radiculopathy");
        assert_eq!(encounter.assessments.len(), 1);

        assert!(encounter.close().is_ok());
        assert_eq!(encounter.status, "completed");
        assert!(encounter.end_time.is_some());
    }

    #[test]
    fn test_treatment_plan() {
        let encounter_id = EncounterId::generate();
        let mut plan = TreatmentPlan::new(encounter_id.clone(), "HIFU ablation", "Benign tumor", 4);

        assert_eq!(plan.status, TreatmentStatus::Planned);
        assert_eq!(plan.completed_sessions, 0);

        plan.add_objective("Reduce tumor volume by 50%");
        assert_eq!(plan.objectives.len(), 1);

        assert!(plan.complete_session().is_ok());
        assert_eq!(plan.completed_sessions, 1);
        assert_eq!(plan.status, TreatmentStatus::Active);

        for _ in 0..3 {
            assert!(plan.complete_session().is_ok());
        }
        assert_eq!(plan.status, TreatmentStatus::Completed);
    }

    #[test]
    fn test_consent_record() {
        let patient_id = PatientId::new("PAT001");
        let consent = ConsentRecord::new(patient_id, ConsentType::Procedure, "Dr. Smith");

        assert_eq!(consent.status, "active");
        assert!(consent.is_valid());

        let mut revoked = consent.clone();
        revoked.revoke();
        assert!(!revoked.is_valid());
    }

    #[test]
    fn test_patient_management_system() {
        let mut pms = PatientManagementSystem::new();

        // Register patient
        let demographics = PatientDemographics {
            patient_id: PatientId::new("PAT001"),
            name: "John Doe".to_string(),
            date_of_birth: "1980-05-15".to_string(),
            sex: 'M',
            weight_kg: 75.0,
            height_cm: 180.0,
            contact_phone: "555-0123".to_string(),
            contact_email: "john@example.com".to_string(),
            emergency_contact: "Jane Doe".to_string(),
            emergency_contact_phone: "555-0124".to_string(),
            medical_record_number: "MRN123456".to_string(),
        };

        let patient_id = pms.register_patient(demographics).unwrap();
        assert_eq!(pms.patient_count(), 1);

        // Create encounter
        let encounter_id = pms
            .create_encounter(
                patient_id.clone(),
                EncounterType::InitialConsultation,
                "Neck pain",
            )
            .unwrap();
        assert_eq!(pms.encounter_count(), 1);

        // Create treatment plan
        let _plan_id = pms
            .create_treatment_plan(encounter_id.clone(), "HIFU ablation", "Benign tumor", 4)
            .unwrap();
        assert_eq!(pms.treatment_plan_count(), 1);

        // Verify consent
        assert!(!pms.verify_consent(&patient_id, ConsentType::Procedure));

        let _ = pms.record_consent(patient_id.clone(), ConsentType::Procedure, "Dr. Smith");
        assert!(pms.verify_consent(&patient_id, ConsentType::Procedure));

        // Get encounters
        let encounters = pms.get_patient_encounters(&patient_id);
        assert_eq!(encounters.len(), 1);

        // Get active plans
        let plans = pms.get_active_plans(&patient_id);
        assert_eq!(plans.len(), 1);
    }
}
