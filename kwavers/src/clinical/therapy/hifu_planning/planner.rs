use super::types::{
    AblationTarget, FocalSpot, HIFUTransducer, HIFUTreatmentPlan, ThermalDose, TreatmentFeasibility,
};
use crate::clinical::therapy::parameters::TherapyParameters;
use crate::core::error::KwaversResult;

/// HIFU Treatment Planner.
#[derive(Debug)]
pub struct HIFUPlanner {
    transducer: HIFUTransducer,
}

impl HIFUPlanner {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
    pub fn new(transducer: HIFUTransducer) -> Self {
        Self { transducer }
    }
    /// Plan treatment.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn plan_treatment(
        &self,
        target: AblationTarget,
        therapy_params: &TherapyParameters,
    ) -> KwaversResult<HIFUTreatmentPlan> {
        let focal_spot = FocalSpot::estimate_from_transducer(&self.transducer);
        let frequency = if therapy_params.frequency > 0.0 {
            therapy_params.frequency
        } else {
            self.transducer.frequency
        };
        let thermal_dose = ThermalDose::estimate_from_focal_spot(
            &focal_spot,
            frequency,
            therapy_params.duty_cycle,
            therapy_params.treatment_duration,
        );
        let mut feasibility = TreatmentFeasibility::new();
        feasibility.focal_coverage_adequate = target.is_focal_spot_adequate(&focal_spot);
        if !feasibility.focal_coverage_adequate {
            feasibility
                .issues
                .push("Focal spot may not adequately cover target".to_owned());
        }
        feasibility.mi_within_limits = focal_spot.is_safe(target.tissue_type);
        if !feasibility.mi_within_limits {
            feasibility.issues.push(format!(
                "MI {:.2} exceeds tissue safety limit {:.2}",
                focal_spot.mechanical_index,
                target.tissue_type.safety_limit()
            ));
        }
        feasibility.thermal_dose_achievable = thermal_dose.is_sufficient_for_ablation();
        if !feasibility.thermal_dose_achievable {
            feasibility.issues.push(format!(
                "Thermal dose {:.0} CEM43 below ablation threshold 240",
                thermal_dose.cem43
            ));
        }
        feasibility.access_path_clear = true;
        feasibility.update_feasibility();
        Ok(HIFUTreatmentPlan {
            transducer: self.transducer.clone(),
            focal_spot,
            target,
            therapy_params: *therapy_params,
            thermal_dose,
            feasibility,
        })
    }

    #[must_use] 
    pub fn transducer(&self) -> &HIFUTransducer {
        &self.transducer
    }
}
