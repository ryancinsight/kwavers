//! Treatment planner core orchestration

use super::types::{SafetyConstraints, TargetVolume, TransducerSpecification, TreatmentPlan};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use log::info;
use ndarray::Array3;

/// Treatment planner for tFUS procedures
#[derive(Debug)]
pub struct TreatmentPlanner {
    /// Computational grid for brain volume
    pub(crate) brain_grid: Grid,
    /// Skull CT data
    pub(crate) skull_ct: Array3<f64>,
    /// Aberration correction calculator
    pub(crate) _aberration_corrector:
        crate::physics::transcranial::aberration_correction::TranscranialAberrationCorrection,
}

impl TreatmentPlanner {
    /// Create new treatment planner
    pub fn new(brain_grid: &Grid, skull_ct_data: &Array3<f64>) -> KwaversResult<Self> {
        let aberration_corrector = crate::physics::transcranial::aberration_correction::TranscranialAberrationCorrection::new(brain_grid)?;

        Ok(Self {
            brain_grid: brain_grid.clone(),
            skull_ct: skull_ct_data.clone(),
            _aberration_corrector: aberration_corrector,
        })
    }

    /// Generate treatment plan for target volumes
    pub fn generate_plan(
        &self,
        patient_id: &str,
        targets: &[TargetVolume],
        transducer_spec: &TransducerSpecification,
    ) -> KwaversResult<TreatmentPlan> {
        info!("Generating tFUS treatment plan for patient: {}", patient_id);
        info!("Planning for {} target volumes", targets.len());

        // Step 1: Analyze skull properties
        let _skull_properties = self.analyze_skull_properties()?;

        // Step 2: Calculate optimal transducer configuration
        let transducer_setup = self.optimize_transducer_setup(targets, transducer_spec)?;

        // Step 3: Simulate acoustic field through skull
        let acoustic_field = self.simulate_acoustic_field(&transducer_setup)?;

        // Step 4: Calculate thermal response
        let temperature_field = self.calculate_thermal_response(&acoustic_field)?;

        // Step 5: Validate safety constraints
        self.validate_safety(
            &temperature_field,
            &acoustic_field,
            transducer_spec.frequency,
        )?;

        // Step 6: Estimate treatment time
        let treatment_time = self.estimate_treatment_time(targets, &acoustic_field);

        Ok(TreatmentPlan {
            patient_id: patient_id.to_string(),
            targets: targets.to_vec(),
            skull_ct: self.skull_ct.clone(),
            transducer_setup,
            acoustic_field,
            temperature_field,
            safety_constraints: SafetyConstraints::default(),
            treatment_time,
        })
    }
}
