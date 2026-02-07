//! Concrete implementation of PinnBeamformingProvider for Burn-based PINNs.
//!
//! This module provides the bridge between the solver layer's Burn PINN implementations
//! and the solver interface's PinnBeamformingProvider trait.
//!
//! ## Architecture
//!
//! ```text
//! Analysis Layer (Layer 4)
//!     ↓ uses trait
//! solver::interface::pinn_beamforming (trait definition)
//!     ↑ implements
//! BurnPinnBeamformingAdapter (this module)
//!     ↓ wraps
//! Burn PINN Implementations (Solver Layer 2)
//! ```

use crate::solver::interface::pinn_beamforming::{
    ModelInfo, PinnBeamformingConfig, PinnBeamformingProvider, PinnBeamformingResult,
    TrainingMetrics, UncertaintyConfig,
};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::sync::{Arc, Mutex};

#[cfg(test)]
use crate::solver::inverse::pinn::ml::BurnLossWeights;
use crate::solver::inverse::pinn::ml::{BurnPINN1DWave, BurnPINNConfig};

/// Adapter that implements PinnBeamformingProvider using Burn-based PINNs.
///
/// This adapter maintains the Burn PINN model internally and translates between
/// the solver-agnostic interface types and Burn-specific types.
impl<B: burn::tensor::backend::Backend> std::fmt::Debug for BurnPinnBeamformingAdapter<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BurnPinnBeamformingAdapter")
            .field("config", &self.config)
            .field("is_trained", &self.is_trained)
            .field("metadata", &self.metadata)
            .field("model", &"<Arc<Mutex<Option<BurnPINN1DWave>>>>")
            .field("device", &"<B::Device>")
            .finish()
    }
}

pub struct BurnPinnBeamformingAdapter<B: burn::tensor::backend::Backend> {
    /// Underlying Burn PINN model
    model: Arc<Mutex<Option<BurnPINN1DWave<B>>>>,
    /// Model configuration
    config: BurnPINNConfig,
    /// Backend device
    device: B::Device,
    /// Is model trained?
    is_trained: bool,
    /// Model metadata
    metadata: ModelInfo,
}

impl<B: burn::tensor::backend::Backend> BurnPinnBeamformingAdapter<B> {
    /// Create new adapter with the given configuration.
    pub fn new(config: BurnPINNConfig, device: B::Device) -> KwaversResult<Self> {
        let metadata = ModelInfo {
            name: "Burn PINN 1D Wave Beamformer".to_string(),
            version: "1.0.0".to_string(),
            num_parameters: Self::estimate_parameters(&config),
            dimensions: vec![1], // 1D wave equation
            is_trained: false,
        };

        Ok(Self {
            model: Arc::new(Mutex::new(None)),
            config,
            device,
            is_trained: false,
            metadata,
        })
    }

    /// Estimate number of parameters in the model.
    fn estimate_parameters(config: &BurnPINNConfig) -> usize {
        // Estimate based on typical PINN architecture: (x,t) -> hidden layers -> u
        let input_size = 2; // Spatiotemporal coordinates
        let output_size = 1; // Scalar field

        if config.hidden_layers.is_empty() {
            return input_size * output_size;
        }

        // Input to first hidden layer
        let mut params = input_size * config.hidden_layers[0] + config.hidden_layers[0];

        // Hidden layers
        for i in 0..config.hidden_layers.len().saturating_sub(1) {
            params +=
                config.hidden_layers[i] * config.hidden_layers[i + 1] + config.hidden_layers[i + 1];
            // weights + biases
        }

        // Last hidden to output
        params += config.hidden_layers[config.hidden_layers.len() - 1] * output_size + output_size;

        params
    }

    /// Initialize the PINN model if not already created.
    fn ensure_model_initialized(&self) -> KwaversResult<()> {
        let mut model_lock = self
            .model
            .lock()
            .map_err(|e| KwaversError::InternalError(format!("Failed to lock model: {}", e)))?;

        if model_lock.is_none() {
            let model = BurnPINN1DWave::new(self.config.clone(), &self.device)?;
            *model_lock = Some(model);
        }

        Ok(())
    }
}

impl<B> PinnBeamformingProvider for BurnPinnBeamformingAdapter<B>
where
    B: burn::tensor::backend::Backend + 'static,
{
    fn beamform(
        &self,
        _rf_data: &Array3<f32>,
        _config: &PinnBeamformingConfig,
    ) -> KwaversResult<PinnBeamformingResult> {
        self.ensure_model_initialized()?;

        Err(KwaversError::NotImplemented(
            "PINN-based beamforming inference not yet implemented. \
             Model architecture exists but forward-pass integration with \
             Burn autodiff backend is pending."
                .into(),
        ))
    }

    fn train(
        &mut self,
        _training_data: &[(Array3<f32>, Array3<f32>)],
        _config: &PinnBeamformingConfig,
    ) -> KwaversResult<TrainingMetrics> {
        self.ensure_model_initialized()?;

        Err(KwaversError::NotImplemented(
            "PINN beamforming training loop not yet implemented. \
             Model and optimizer infrastructure exists but gradient \
             computation via Burn autodiff is pending."
                .into(),
        ))
    }

    fn estimate_uncertainty(
        &self,
        rf_data: &Array3<f32>,
        config: &UncertaintyConfig,
    ) -> KwaversResult<Array3<f32>> {
        self.ensure_model_initialized()?;

        if config.bayesian_enabled {
            // Monte Carlo dropout for Bayesian uncertainty estimation
            // Requires trained PINN model with dropout layers
            return Err(KwaversError::NotImplemented(
                "Bayesian uncertainty estimation via MC dropout not yet implemented. \
                 Requires trained PINN model with dropout-based stochastic inference."
                    .into(),
            ));
        }

        // Simple signal-based uncertainty (higher for weaker signals)
        Ok(rf_data.mapv(|v| 1.0 / (v.abs() + 1.0)))
    }

    fn is_ready(&self) -> bool {
        self.is_trained
    }

    fn model_info(&self) -> ModelInfo {
        self.metadata.clone()
    }
}

/// Factory function to create a Burn PINN beamforming provider.
#[cfg(feature = "pinn")]
pub fn create_burn_beamforming_provider() -> KwaversResult<Box<dyn PinnBeamformingProvider>> {
    type Backend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    let config = BurnPINNConfig::default();
    let device = Default::default();

    let adapter = BurnPinnBeamformingAdapter::<Backend>::new(config, device)?;
    Ok(Box::new(adapter))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        type Backend = burn::backend::NdArray<f32>;
        let config = BurnPINNConfig::default();
        let device = Default::default();

        let result = BurnPinnBeamformingAdapter::<Backend>::new(config, device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parameter_estimation() {
        let config = BurnPINNConfig {
            hidden_layers: vec![64, 64],
            learning_rate: 1e-3,
            loss_weights: BurnLossWeights::default(),
            num_collocation_points: 10_000,
        };

        let params =
            BurnPinnBeamformingAdapter::<burn::backend::NdArray<f32>>::estimate_parameters(&config);

        // Expected: (2*64+64) + (64*64+64) + (64*1+1) = 192 + 4160 + 65 = 4417
        assert_eq!(params, 4417);
    }

    #[test]
    fn test_model_info() {
        type Backend = burn::backend::NdArray<f32>;
        let config = BurnPINNConfig::default();
        let device = Default::default();

        let adapter = BurnPinnBeamformingAdapter::<Backend>::new(config, device).unwrap();
        let info = adapter.model_info();

        assert_eq!(info.name, "Burn PINN 1D Wave Beamformer");
        assert!(!info.is_trained);
        assert_eq!(info.dimensions, vec![1]);
        assert!(info.num_parameters > 0);
    }
}
