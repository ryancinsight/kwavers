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

use crate::core::error::{KwaversError, KwaversResult, SystemError};
use crate::solver::interface::pinn_beamforming::{
    ModelInfo, PinnBeamformingConfig, PinnBeamformingProvider, NeuralPinnBeamformingResult,
    PinnBeamformingUncertaintyConfig, TrainingMetrics,
};
use burn::tensor::backend::AutodiffBackend;
use ndarray::{Array1, Array2, Array3};
use std::sync::{Arc, Mutex};

#[cfg(test)]
use crate::solver::inverse::pinn::ml::BurnLossWeights;
use crate::solver::inverse::pinn::ml::{BurnPINN1DWave, BurnPINNConfig, BurnPINNTrainer};

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    B: AutodiffBackend + 'static,
{
    fn beamform(
        &self,
        rf_data: &Array3<f32>,
        _config: &PinnBeamformingConfig,
    ) -> KwaversResult<NeuralPinnBeamformingResult> {
        use std::time::Instant;
        let t_start = Instant::now();

        self.ensure_model_initialized()?;

        let (n_channels, n_samples, n_frames) = rf_data.dim();
        if n_channels == 0 || n_samples == 0 {
            return Err(KwaversError::InvalidInput(
                "RF data must have non-zero channel and sample dimensions".into(),
            ));
        }

        // Build (x, t) coordinate grid: channel → x ∈ [-1, 1], sample → t ∈ [0, 1].
        let n_points = n_channels * n_samples;
        let x_scale = 1.0 / (n_channels.saturating_sub(1).max(1)) as f64;
        let t_scale = 1.0 / (n_samples.saturating_sub(1).max(1)) as f64;
        let mut x_flat = Array1::<f64>::zeros(n_points);
        let mut t_flat = Array1::<f64>::zeros(n_points);
        for si in 0..n_samples {
            for ci in 0..n_channels {
                let idx = si * n_channels + ci;
                x_flat[idx] = ci as f64 * x_scale * 2.0 - 1.0;
                t_flat[idx] = si as f64 * t_scale;
            }
        }

        let predictions = {
            let model_lock = self
                .model
                .lock()
                .map_err(|e| KwaversError::InternalError(format!("Failed to lock model: {e}")))?;
            match model_lock.as_ref() {
                Some(model) => model.predict(&x_flat, &t_flat, &self.device)?,
                None => return Err(KwaversError::InternalError("Model not initialized".into())),
            }
        };

        // Map predictions [n_points, 1] → [n_channels, n_samples, n_frames].
        // All frames receive the same PINN prediction (time-independent inference
        // over the frame dimension — the frame axis is a recording repetition axis,
        // not a new coordinate).
        let mut image = Array3::<f32>::zeros((n_channels, n_samples, n_frames));
        for si in 0..n_samples {
            for ci in 0..n_channels {
                let val = predictions[[si * n_channels + ci, 0]] as f32;
                for fi in 0..n_frames {
                    image[[ci, si, fi]] = val;
                }
            }
        }

        Ok(NeuralPinnBeamformingResult {
            image,
            uncertainty: None,
            confidence: None,
            inference_time: t_start.elapsed().as_secs_f64(),
            metrics: TrainingMetrics {
                total_loss: 0.0,
                physics_loss: 0.0,
                data_loss: 0.0,
                iterations: 0,
                training_time: 0.0,
            },
        })
    }

    fn train(
        &mut self,
        training_data: &[(Array3<f32>, Array3<f32>)],
        _config: &PinnBeamformingConfig,
    ) -> KwaversResult<TrainingMetrics> {
        self.ensure_model_initialized()?;

        if training_data.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Training data must be non-empty".into(),
            ));
        }

        // Flatten all target frames into a combined (x, t, u) dataset.
        // RF layout: (n_channels, n_samples, n_frames).
        // x ∈ [-1, 1]: normalised channel index; t ∈ [0, 1]: normalised sample index.
        let mut x_vals: Vec<f64> = Vec::new();
        let mut t_vals: Vec<f64> = Vec::new();
        let mut u_vals: Vec<f64> = Vec::new();

        for (_, target) in training_data {
            let (n_ch, n_sa, n_fr) = target.dim();
            let x_scale = 1.0 / (n_ch.saturating_sub(1).max(1)) as f64;
            let t_scale = 1.0 / (n_sa.saturating_sub(1).max(1)) as f64;
            for fi in 0..n_fr {
                for si in 0..n_sa {
                    for ci in 0..n_ch {
                        x_vals.push(ci as f64 * x_scale * 2.0 - 1.0);
                        t_vals.push(si as f64 * t_scale);
                        u_vals.push(target[[ci, si, fi]] as f64);
                    }
                }
            }
        }

        let n = x_vals.len();
        let x_data = Array1::from_vec(x_vals);
        let t_data = Array1::from_vec(t_vals);
        let u_data = Array2::from_shape_vec((n, 1), u_vals)
            .map_err(|e| KwaversError::InvalidInput(format!("u_data reshape: {e}")))?;

        // 1000 epochs by default; wave speed 1500 m/s (soft tissue).
        let epochs = 1000_usize;
        let wave_speed = 1500.0_f64;

        let mut trainer = BurnPINNTrainer::<B>::new(self.config.clone(), &self.device)?;
        let burn_metrics =
            trainer.train(&x_data, &t_data, &u_data, wave_speed, &self.device, epochs)?;

        // Store trained model; release lock before mutating self.
        {
            let mut model_lock = self
                .model
                .lock()
                .map_err(|e| KwaversError::InternalError(format!("Failed to lock model: {e}")))?;
            *model_lock = Some(trainer.pinn().clone());
        }

        self.is_trained = true;
        self.metadata.is_trained = true;

        Ok(TrainingMetrics {
            total_loss: burn_metrics.total_loss.last().copied().unwrap_or(0.0),
            physics_loss: burn_metrics.pde_loss.last().copied().unwrap_or(0.0),
            data_loss: burn_metrics.data_loss.last().copied().unwrap_or(0.0),
            iterations: burn_metrics.epochs_completed,
            training_time: burn_metrics.training_time_secs,
        })
    }

    fn estimate_uncertainty(
        &self,
        rf_data: &Array3<f32>,
        config: &PinnBeamformingUncertaintyConfig,
    ) -> KwaversResult<Array3<f32>> {
        self.ensure_model_initialized()?;

        if config.bayesian_enabled {
            // MC dropout requires stochastic PINN layers not present in the current architecture.
            return Err(KwaversError::System(SystemError::FeatureNotAvailable {
                feature: "Bayesian MC-dropout uncertainty".to_string(),
                reason: "Current PINN architecture has no dropout layers; \
                         add stochastic dropout to the network and retrain."
                    .to_string(),
            }));
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
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
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
        let _adapter = result.unwrap();
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
        // AutodiffBackend required by PinnBeamformingProvider impl bound.
        type Backend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
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
