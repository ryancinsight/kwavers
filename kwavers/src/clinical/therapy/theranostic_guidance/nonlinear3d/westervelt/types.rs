use ndarray::Array3;

use super::super::types::VolumeReconstructionMetrics;

#[derive(Clone, Debug)]
pub(crate) struct WesterveltFwiResult {
    pub reconstructed_sound_speed_m_s: Array3<f64>,
    pub reconstructed_delta_c_m_s: Array3<f64>,
    pub reconstructed_beta: Array3<f64>,
    pub reconstructed_delta_beta: Array3<f64>,
    pub multiparameter_fwi_score: Array3<f64>,
    pub peak_pressure_pa: Array3<f64>,
    pub objective_history: Vec<f64>,
    pub metrics: VolumeReconstructionMetrics,
    pub dt_s: f64,
    pub time_steps: usize,
}
