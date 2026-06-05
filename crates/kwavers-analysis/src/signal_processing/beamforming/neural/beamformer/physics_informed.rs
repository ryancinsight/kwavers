use ndarray::Array4;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::features;
use super::super::types::HybridBeamformingResult;
use super::NeuralBeamformer;

impl NeuralBeamformer {
    /// Process physics informed.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn process_physics_informed(
        &self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<HybridBeamformingResult> {
        let network = self
            .neural_network
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Neural network not initialized".into()))?;

        let base_image = self.traditional_beamforming(rf_data, steering_angles)?;
        let feat = features::extract_all_features(&base_image);

        let network_output =
            network.forward_physics_informed(&feat, steering_angles, &self.physics_constraints)?;

        let scale_factor = network_output[[0, 0, 0]];
        let beamformed = &base_image * scale_factor;

        let uncertainty = self.uncertainty_estimator.estimate(&beamformed)?;
        let mean_uncertainty = uncertainty.mean().unwrap_or(0.0) as f64;

        Ok(HybridBeamformingResult {
            image: beamformed,
            uncertainty: Some(uncertainty),
            confidence: 0.95 - mean_uncertainty * 0.05,
            processing_mode: "PhysicsInformed".to_string(),
        })
    }
}
