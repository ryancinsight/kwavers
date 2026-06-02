//! PINN inference engine trait for neural beamforming.

use kwavers_core::error::KwaversResult;

pub trait PinnInferenceEngine: Send + Sync {
    /// Run real-time PINN inference at the given spatial-temporal coordinates.
    ///
    /// # Errors
    /// - Returns [`Err`] if model inference fails or coordinate arrays have mismatched length.
    fn predict_realtime(
        &mut self,
        x_coords: &[f32],
        y_coords: &[f32],
        t_coords: &[f32],
    ) -> KwaversResult<(Vec<f32>, Vec<f32>)>;
}
