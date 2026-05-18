use super::{ClinicalSafetyLimits, SafetyViolation};
use crate::clinical::therapy::parameters::ClinicalTherapyParameters;
use crate::core::error::{KwaversError, KwaversResult};
use std::time::Instant;

/// Treatment dose controller with IEC compliance
#[derive(Debug)]
pub struct DoseController {
    pub(super) safety_limits: ClinicalSafetyLimits,
    pub accumulated_dose: f64,
    session_start_time: Option<Instant>,
    treatment_history: Vec<TreatmentRecord>,
}

impl DoseController {
    /// Create new dose controller.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(safety_limits: ClinicalSafetyLimits) -> Self {
        Self {
            safety_limits,
            accumulated_dose: 0.0,
            session_start_time: None,
            treatment_history: Vec::new(),
        }
    }

    /// Start new treatment session.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn start_session(&mut self, patient_id: String, protocol: String) -> KwaversResult<()> {
        if self.accumulated_dose >= self.safety_limits.max_total_dose {
            return Err(KwaversError::InvalidInput(
                "Cannot start session: maximum total dose already reached".to_owned(),
            ));
        }

        self.session_start_time = Some(Instant::now());
        self.treatment_history.push(TreatmentRecord {
            patient_id,
            protocol,
            start_time: Instant::now(),
            end_time: None,
            delivered_dose: 0.0,
            safety_violations: Vec::new(),
        });

        log::info!("Treatment session started");
        Ok(())
    }

    /// Update delivered dose during treatment.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn update_dose(
        &mut self,
        incremental_dose: f64,
        params: &ClinicalTherapyParameters,
    ) -> KwaversResult<()> {
        if let Some(start_time) = self.session_start_time {
            let session_duration = start_time.elapsed().as_secs_f64();
            if session_duration > self.safety_limits.max_session_time {
                return Err(KwaversError::InvalidInput(
                    "Session time limit exceeded".to_owned(),
                ));
            }
        }

        if self.accumulated_dose + incremental_dose > self.safety_limits.max_total_dose {
            return Err(KwaversError::InvalidInput(
                "Total dose limit would be exceeded".to_owned(),
            ));
        }

        self.accumulated_dose += incremental_dose;

        if let Some(record) = self.treatment_history.last_mut() {
            record.delivered_dose += incremental_dose;
        }

        log::info!(
            "Dose update: +{:.1} J (total: {:.1} J), Pressure: {:.1} Pa, MI: {:.2}",
            incremental_dose,
            self.accumulated_dose,
            params.peak_negative_pressure,
            params.mechanical_index
        );

        Ok(())
    }

    /// End current treatment session.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn end_session(&mut self) -> KwaversResult<()> {
        if let Some(record) = self.treatment_history.last_mut() {
            record.end_time = Some(Instant::now());
        }

        self.session_start_time = None;
        log::info!(
            "Treatment session ended. Total accumulated dose: {:.1} J",
            self.accumulated_dose
        );
        Ok(())
    }

    /// Get remaining dose capacity (J).
    #[must_use]
    pub fn remaining_dose_capacity(&self) -> f64 {
        self.safety_limits.max_total_dose - self.accumulated_dose
    }

    /// Get treatment history.
    #[must_use]
    pub fn treatment_history(&self) -> &[TreatmentRecord] {
        &self.treatment_history
    }

    /// Reset accumulated dose for a new patient or treatment course.
    pub fn reset_accumulated_dose(&mut self) {
        self.accumulated_dose = 0.0;
        log::warn!("Accumulated dose reset - new patient/course");
    }
}

/// Treatment record for audit trail
#[derive(Debug, Clone)]
pub struct TreatmentRecord {
    pub patient_id: String,
    pub protocol: String,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub delivered_dose: f64,
    pub safety_violations: Vec<SafetyViolation>,
}
