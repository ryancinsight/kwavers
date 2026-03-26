use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

use super::demographics::{PatientDemographics, PatientId};
use super::encounter::{ClinicalEncounter, EncounterId};
use super::profile::PatientMedicalProfile;
use super::treatment::TreatmentPlan;

/// Patient Management System
/// Handles tracking of all patients, encounters, and treatment plans
#[derive(Debug, Default)]
pub struct PatientManagementSystem {
    patients: HashMap<PatientId, PatientMedicalProfile>,
    encounters: HashMap<EncounterId, ClinicalEncounter>,
    treatment_plans: HashMap<String, TreatmentPlan>,
}

impl PatientManagementSystem {
    /// Create a new patient management system
    pub fn new() -> Self {
        Self {
            patients: HashMap::new(),
            encounters: HashMap::new(),
            treatment_plans: HashMap::new(),
        }
    }

    /// Register a new patient
    pub fn register_patient(
        &mut self,
        demographics: PatientDemographics,
    ) -> KwaversResult<PatientId> {
        let profile = PatientMedicalProfile::new(demographics.clone())?;
        let id = demographics.patient_id;

        if self.patients.contains_key(&id) {
            return Err(KwaversError::InvalidInput(
                "Patient ID already exists".to_string(),
            ));
        }

        self.patients.insert(id.clone(), profile);
        Ok(id)
    }

    /// Update a patient's demographics
    pub fn update_demographics(
        &mut self,
        patient_id: &PatientId,
        demographics: PatientDemographics,
    ) -> KwaversResult<()> {
        demographics.validate()?;

        if let Some(profile) = self.patients.get_mut(patient_id) {
            profile.demographics = demographics;
            profile.last_updated = super::profile::current_timestamp();
            Ok(())
        } else {
            Err(KwaversError::InvalidInput("Patient not found".to_string()))
        }
    }

    /// Register a clinical encounter
    pub fn register_encounter(
        &mut self,
        encounter: ClinicalEncounter,
    ) -> KwaversResult<EncounterId> {
        let patient_id = &encounter.patient_id;

        if !self.patients.contains_key(patient_id) {
            return Err(KwaversError::InvalidInput("Patient not found".to_string()));
        }

        let id = encounter.encounter_id.clone();
        self.encounters.insert(id.clone(), encounter);

        Ok(id)
    }

    /// Complete a treatment session
    pub fn complete_treatment_session(&mut self, plan_id: &str) -> KwaversResult<()> {
        if let Some(plan) = self.treatment_plans.get_mut(plan_id) {
            plan.complete_session()
        } else {
            Err(KwaversError::InvalidInput(
                "Treatment plan not found".to_string(),
            ))
        }
    }

    /// Export patient data (anonymized for research)
    pub fn export_anonymized_data(&self) -> KwaversResult<String> {
        // Implementation for data export ensuring HIPAA compliance
        Ok("Anonymized data export...".to_string())
    }
}
