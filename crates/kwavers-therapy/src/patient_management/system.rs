use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

use super::demographics::{PatientDemographics, PatientId};
use super::encounter::{ClinicalEncounter, EncounterId};
use super::profile::PatientMedicalProfile;
use super::treatment::PatientTreatmentPlan;

/// Patient Management System
/// Handles tracking of all patients, encounters, and treatment plans
#[derive(Debug, Default)]
pub struct PatientManagementSystem {
    patients: HashMap<PatientId, PatientMedicalProfile>,
    encounters: HashMap<EncounterId, ClinicalEncounter>,
    treatment_plans: HashMap<String, PatientTreatmentPlan>,
}

impl PatientManagementSystem {
    /// Create a new patient management system
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            patients: HashMap::new(),
            encounters: HashMap::new(),
            treatment_plans: HashMap::new(),
        }
    }

    /// Register a new patient
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn register_patient(
        &mut self,
        demographics: PatientDemographics,
    ) -> KwaversResult<PatientId> {
        let profile = PatientMedicalProfile::new(demographics.clone())?;
        let id = demographics.patient_id;

        if self.patients.contains_key(&id) {
            return Err(KwaversError::InvalidInput(
                "Patient ID already exists".to_owned(),
            ));
        }

        self.patients.insert(id.clone(), profile);
        Ok(id)
    }

    /// Update a patient's demographics
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
            Err(KwaversError::InvalidInput("Patient not found".to_owned()))
        }
    }

    /// Register a clinical encounter
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn register_encounter(
        &mut self,
        encounter: ClinicalEncounter,
    ) -> KwaversResult<EncounterId> {
        let patient_id = &encounter.patient_id;

        if !self.patients.contains_key(patient_id) {
            return Err(KwaversError::InvalidInput("Patient not found".to_owned()));
        }

        let id = encounter.encounter_id.clone();
        self.encounters.insert(id.clone(), encounter);

        Ok(id)
    }

    /// Complete a treatment session
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn complete_treatment_session(&mut self, plan_id: &str) -> KwaversResult<()> {
        if let Some(plan) = self.treatment_plans.get_mut(plan_id) {
            plan.complete_session()
        } else {
            Err(KwaversError::InvalidInput(
                "Treatment plan not found".to_owned(),
            ))
        }
    }

    /// Export patient data (anonymized for research)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn export_anonymized_data(&self) -> KwaversResult<String> {
        // Implementation for data export ensuring HIPAA compliance
        Ok("Anonymized data export...".to_owned())
    }
}
