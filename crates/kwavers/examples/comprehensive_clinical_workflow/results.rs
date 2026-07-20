//! Result types emitted by the liver-assessment workflow.

use kwavers_analysis::ml::uncertainty::BeamformingUncertainty;
use kwavers_imaging::ultrasound::elastography::NonlinearParameterMap;
use kwavers_solver::forward::elastic::ElasticWaveField;
use leto::{Array3, Array4};
use std::time::Duration;

#[derive(Debug)]
pub(super) struct LiverAssessmentReport {
    pub(super) patient_id: String,
    pub(super) b_mode_result: BModeResult,
    pub(super) swe_result: SWEResult,
    pub(super) ceus_result: CEUSResult,
    pub(super) uncertainty_result: UncertaintyAnalysis,
    pub(super) diagnosis: ClinicalDiagnosis,
    pub(super) treatment_plan: TreatmentPlan,
    pub(super) safety_assessment: SafetyAssessment,
    pub(super) validation_report: String,
    pub(super) processing_time: Duration,
}

#[derive(Debug)]
pub(super) struct BModeResult {
    pub(super) envelope: Array3<f32>,
    pub(super) axial_resolution: f64,
    pub(super) lateral_resolution: f64,
    pub(super) contrast_ratio: f64,
}

#[derive(Debug)]
pub(super) struct SWEResult {
    pub(super) stiffness_map: Array3<f32>,
    pub(super) displacement_history: Vec<ElasticWaveField>,
    pub(super) nonlinear_analysis: NonlinearParameterMap,
    pub(super) fibrosis_metrics: FibrosisMetrics,
}

#[derive(Debug)]
pub(super) struct CEUSResult {
    pub(super) contrast_signal: Array4<f32>,
    pub(super) perfusion_map: Array3<f32>,
    pub(super) perfusion_metrics: PerfusionMetrics,
}

#[derive(Debug)]
pub(super) struct UncertaintyAnalysis {
    pub(super) swe_uncertainty: BeamformingUncertainty,
    pub(super) perfusion_uncertainty: BeamformingUncertainty,
}

#[derive(Debug)]
pub(super) struct ClinicalDiagnosis {
    pub(super) diagnosis_text: String,
    pub(super) fibrosis_stage: i32,
    pub(super) perfusion_status: String,
    pub(super) confidence_level: f64,
    pub(super) recommendations: Vec<String>,
}

#[derive(Debug)]
pub(super) struct TreatmentPlan {
    pub(super) recommended_actions: String,
    pub(super) follow_up_schedule: String,
    pub(super) additional_tests: Vec<String>,
    pub(super) therapeutic_considerations: String,
}

#[derive(Debug)]
pub(super) struct SafetyAssessment {
    pub(super) acoustic_safety: bool,
    pub(super) thermal_safety: bool,
    pub(super) overall_safe: bool,
    pub(super) safety_notes: String,
}

#[derive(Debug)]
pub(super) struct FibrosisMetrics {
    pub(super) mean_stiffness: f64,
    pub(super) stiffness_std: f64,
    pub(super) fibrosis_stage: i32,
    pub(super) nonlinear_parameter: f64,
}

#[derive(Debug)]
pub(super) struct PerfusionMetrics {
    pub(super) peak_enhancement: f64,
    pub(super) perfusion_rate: f64,
    pub(super) wash_in_time: f64,
    pub(super) wash_out_time: f64,
}
