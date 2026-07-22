use leto::Array4;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::config::NeuralBeamformingMode;
use super::super::features;
use super::super::types::HybridBeamformingResult;
use super::NeuralBeamformer;

impl NeuralBeamformer {
    /// Process.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn process(
        &mut self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<HybridBeamformingResult> {
        let start_time = std::time::Instant::now();

        let result = match self.config.mode {
            NeuralBeamformingMode::NeuralOnly => {
                self.process_neural_only(rf_data, steering_angles)?
            }
            NeuralBeamformingMode::Hybrid => self.process_hybrid(rf_data, steering_angles)?,
            #[cfg(feature = "pinn")]
            NeuralBeamformingMode::PhysicsInformed => {
                self.process_physics_informed(rf_data, steering_angles)?
            }
            NeuralBeamformingMode::Adaptive => self.process_adaptive(rf_data, steering_angles)?,
        };

        let processing_time = start_time.elapsed().as_secs_f64();
        self.metrics.update(processing_time, result.confidence);

        Ok(result)
    }
    /// Process neural only.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub(super) fn process_neural_only(
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
        let network_output = network.forward(&feat, steering_angles)?;

        let scale_factor = network_output[[0, 0, 0]];
        let beamformed = &base_image * scale_factor;

        let uncertainty = self.uncertainty_estimator.estimate(&beamformed)?;
        let mean_uncertainty = leto::mean_all(&uncertainty).unwrap_or(0.0) as f64;

        Ok(HybridBeamformingResult {
            image: beamformed,
            uncertainty: Some(uncertainty),
            confidence: 1.0 - mean_uncertainty,
            processing_mode: "NeuralOnly".to_owned(),
        })
    }
    /// Process hybrid.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub(super) fn process_hybrid(
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
        let network_output = network.forward(&feat, steering_angles)?;

        let scale_factor = network_output[[0, 0, 0]];
        let refined = &base_image * scale_factor;

        let constrained = self.physics_constraints.apply(&refined)?;
        let uncertainty = self.uncertainty_estimator.estimate(&constrained)?;
        let mean_uncertainty = leto::mean_all(&uncertainty).unwrap_or(0.0) as f64;

        Ok(HybridBeamformingResult {
            image: constrained,
            uncertainty: Some(uncertainty),
            confidence: 0.9 - mean_uncertainty * 0.1,
            processing_mode: "Hybrid".to_owned(),
        })
    }
    /// Process adaptive.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub(super) fn process_adaptive(
        &self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<HybridBeamformingResult> {
        let signal_quality = self.assess_signal_quality(rf_data)?;
        let quality_threshold = self.config.adaptation_parameters.quality_threshold;

        if signal_quality > quality_threshold {
            let mut result = self.process_neural_only(rf_data, steering_angles)?;
            result.processing_mode = "Adaptive(Neural)".to_owned();
            Ok(result)
        } else {
            let mut result = self.process_hybrid(rf_data, steering_angles)?;
            result.processing_mode = "Adaptive(Hybrid)".to_owned();
            Ok(result)
        }
    }
}
