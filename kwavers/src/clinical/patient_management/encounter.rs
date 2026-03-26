use super::demographics::PatientId;
use super::profile::current_timestamp;
use super::treatment::TreatmentPlan;
use crate::core::error::{KwaversError, KwaversResult};

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
#[derive(Debug, Clone, Default)]
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

impl VitalSigns {
    /// Validate vital signs are within reasonable ranges
    pub fn validate(&self) -> KwaversResult<()> {
        if let Some(hr) = self.heart_rate_bpm {
            if !(30..=200).contains(&hr) {
                return Err(KwaversError::InvalidInput(
                    "Heart rate out of reasonable range".to_string(),
                ));
            }
        }

        if let Some((sys, dia)) = self.blood_pressure {
            if !(70..=250).contains(&sys) || !(40..=150).contains(&dia) {
                return Err(KwaversError::InvalidInput(
                    "Blood pressure out of reasonable range".to_string(),
                ));
            }
        }

        if let Some(temp) = self.temperature_celsius {
            if !(35.0..=42.0).contains(&temp) {
                return Err(KwaversError::InvalidInput(
                    "Temperature out of reasonable range".to_string(),
                ));
            }
        }

        if let Some(o2) = self.oxygen_saturation_percent {
            if !(50.0..=100.0).contains(&o2) {
                return Err(KwaversError::InvalidInput(
                    "Oxygen saturation out of range".to_string(),
                ));
            }
        }

        Ok(())
    }
}
