//! Public types for elastic shear FWI reconstruction.

use leto::Array2;

pub const THERANOSTIC_ELASTIC_SHEAR_MODEL: &str =
    "iterative_nonlinear_elastic_pstd_fwi_residual_migration";

#[derive(Clone, Debug)]
pub struct ElasticShearReconstructionResult {
    pub reconstruction: Array2<f64>,
    pub model_name: String,
    pub receiver_count: usize,
    pub time_steps: usize,
    pub dt_s: f64,
    pub iteration_count: usize,
    pub accepted_step_count: usize,
    pub objective_history: Vec<f64>,
    pub baseline_trace_energy: f64,
    pub lesion_trace_energy: f64,
    pub residual_trace_energy: f64,
}

pub(super) struct ElasticFwiInversion {
    pub model: Array2<f64>,
    pub iteration_count: usize,
    pub accepted_step_count: usize,
    pub objective_history: Vec<f64>,
    pub final_residual_energy: f64,
}
