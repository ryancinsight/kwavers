//! PINN inference engine trait for neural beamforming.

use crate::core::error::KwaversResult;

pub trait PinnInferenceEngine: Send + Sync {
    fn predict_realtime(
        &mut self,
        x_coords: &[f32],
        y_coords: &[f32],
        t_coords: &[f32],
    ) -> KwaversResult<(Vec<f32>, Vec<f32>)>;
}
