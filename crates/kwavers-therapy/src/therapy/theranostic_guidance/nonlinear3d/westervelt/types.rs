use leto::Array3;

use super::super::types::{
    FwiIterationDiagnostics, SourcePlanMetrics, VolumeReconstructionMetrics,
};

#[derive(Clone, Debug)]
pub(crate) struct WesterveltFwiResult {
    pub reconstructed_sound_speed_m_s: Array3<f64>,
    pub reconstructed_delta_c_m_s: Array3<f64>,
    pub reconstructed_beta: Array3<f64>,
    pub reconstructed_delta_beta: Array3<f64>,
    pub multiparameter_fwi_score: Array3<f64>,
    pub peak_pressure_pa: Array3<f64>,
    pub objective_history: Vec<f64>,
    pub iteration_diagnostics: Vec<FwiIterationDiagnostics>,
    pub metrics: VolumeReconstructionMetrics,
    pub source_scale: f64,
    pub source_plan_metrics: SourcePlanMetrics,
    pub dt_s: f64,
    pub time_steps: usize,
}
