use super::schedule::SonicationSchedule;
use super::types::{
    AblationTarget, ClinicalHIFUTransducer, ClinicalHIFUTreatmentPlan, FocalSpot,
    FocalSpotDoseEstimate, TreatmentFeasibility,
};
use crate::therapy::domain_types::ClinicalTherapyParameters;
use kwavers_core::error::KwaversResult;

/// HIFU Treatment Planner.
#[derive(Debug)]
pub struct HIFUPlanner {
    transducer: ClinicalHIFUTransducer,
}

impl HIFUPlanner {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(transducer: ClinicalHIFUTransducer) -> Self {
        Self { transducer }
    }
    /// Plan treatment.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn plan_treatment(
        &self,
        target: AblationTarget,
        therapy_params: &ClinicalTherapyParameters,
    ) -> KwaversResult<ClinicalHIFUTreatmentPlan> {
        let focal_spot = FocalSpot::estimate_from_transducer(&self.transducer)?;
        let frequency = if therapy_params.frequency > 0.0 {
            therapy_params.frequency
        } else {
            self.transducer.frequency
        };
        let thermal_dose = FocalSpotDoseEstimate::estimate_from_focal_spot(
            &focal_spot,
            frequency,
            therapy_params.duty_cycle,
            therapy_params.treatment_duration,
        )?;
        let sonication_schedule = self.plan_sonication_schedule(&target, therapy_params)?;
        let mut feasibility = TreatmentFeasibility::new();
        feasibility.focal_coverage_adequate = sonication_schedule.coverage_guaranteed;
        if !feasibility.focal_coverage_adequate {
            feasibility
                .issues
                .push("Subspot grid does not prove full target coverage".to_owned());
        }
        feasibility.mi_within_limits = focal_spot.is_safe(target.tissue_type);
        if !feasibility.mi_within_limits {
            feasibility.issues.push(format!(
                "MI {:.2} exceeds tissue safety limit {:.2}",
                focal_spot.mechanical_index,
                target.tissue_type.safety_limit()
            ));
        }
        feasibility.thermal_dose_achievable = sonication_schedule.all_subspots_reach_ablation();
        if !feasibility.thermal_dose_achievable {
            feasibility.issues.push(format!(
                "Minimum subspot thermal dose {:.0} CEM43 below ablation threshold 240",
                sonication_schedule.minimum_subspot_cem43.as_minutes()
            ));
        }
        feasibility.access_path_clear = true;
        feasibility.update_feasibility();
        Ok(ClinicalHIFUTreatmentPlan {
            transducer: self.transducer.clone(),
            focal_spot,
            target,
            therapy_params: *therapy_params,
            thermal_dose,
            feasibility,
        })
    }

    /// Plan a target-covering sonication schedule without changing the
    /// `ClinicalHIFUTreatmentPlan` struct layout.
    ///
    /// # Errors
    /// Returns [`Err`] when the target dimensions, safety margin, focal widths,
    /// or treatment duration are non-finite or non-positive.
    pub fn plan_sonication_schedule(
        &self,
        target: &AblationTarget,
        therapy_params: &ClinicalTherapyParameters,
    ) -> KwaversResult<SonicationSchedule> {
        let focal_spot = FocalSpot::estimate_from_transducer(&self.transducer)?;
        let frequency = if therapy_params.frequency > 0.0 {
            therapy_params.frequency
        } else {
            self.transducer.frequency
        };
        SonicationSchedule::plan(target, &focal_spot, therapy_params, frequency)
    }

    #[must_use]
    pub fn transducer(&self) -> &ClinicalHIFUTransducer {
        &self.transducer
    }
}
