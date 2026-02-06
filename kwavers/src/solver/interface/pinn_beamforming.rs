//! PINN-Beamforming Interface
//!
//! This module defines the provider traits and associated types for Physics-Informed
//! Neural Network (PINN) based beamforming. It lives in the solver interface layer
//! so that both solver implementations (which implement these traits) and analysis
//! consumers (which use them) can depend on it without cross-layer violations.
//!
//! # Architecture
//!
//! ```text
//! analysis::signal_processing::beamforming  (Layer 4 — consumes trait)
//!     ↓ depends on
//! solver::interface::pinn_beamforming       (Layer 2 — defines trait)
//!     ↑ implemented by
//! solver::inverse::pinn::beamforming        (Layer 2 — concrete impl)
//! ```

use crate::core::error::KwaversResult;
use ndarray::{Array3, Array4};

/// Configuration for PINN-based beamforming.
#[derive(Debug, Clone, Default)]
pub struct PinnBeamformingConfig {
    /// PINN model configuration
    pub model: PinnModelConfig,
    /// Inference configuration
    pub inference: InferenceConfig,
    /// Uncertainty quantification settings
    pub uncertainty: UncertaintyConfig,
}

/// PINN model configuration.
#[derive(Debug, Clone)]
pub struct PinnModelConfig {
    /// Model architecture type
    pub architecture: ModelArchitecture,
    /// Number of hidden layers
    pub hidden_layers: usize,
    /// Neurons per hidden layer
    pub neurons_per_layer: usize,
    /// Activation function
    pub activation: ActivationFunction,
    /// Model version
    pub version: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Supported dimensions (1D, 2D, 3D)
    pub dimensions: Vec<usize>,
    /// Is the model trained?
    pub is_trained: bool,
}

impl Default for PinnModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Mlp,
            hidden_layers: 4,
            neurons_per_layer: 128,
            activation: ActivationFunction::Tanh,
            version: "1.0.0".to_string(),
            num_parameters: 0,
            dimensions: vec![1, 2, 3],
            is_trained: false,
        }
    }
}

/// Model architecture types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// Multi-layer perceptron
    Mlp,
    /// Fourier feature network
    Fourier,
    /// Physics-informed neural operator
    Pino,
    /// Deep operator network
    DeepOnet,
}

/// Activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    /// Tanh activation
    Tanh,
    /// ReLU activation
    Relu,
    /// Sigmoid activation
    Sigmoid,
    /// Swish activation
    Swish,
    /// GELU activation
    Gelu,
}

/// Inference configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Batch size for inference
    pub batch_size: usize,
    /// Use half-precision (FP16)
    pub use_fp16: bool,
    /// Device configuration
    pub device: DeviceConfig,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            use_fp16: false,
            device: DeviceConfig::Cpu,
        }
    }
}

/// Device configuration for inference.
#[derive(Debug, Clone)]
pub enum DeviceConfig {
    /// CPU inference
    Cpu,
    /// GPU inference with device ID
    Gpu {
        /// GPU device ID
        device_id: usize,
    },
}

/// Uncertainty quantification configuration.
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Enable Bayesian inference
    pub bayesian_enabled: bool,
    /// Number of Monte Carlo samples
    pub mc_samples: usize,
    /// Confidence level (0-1)
    pub confidence_level: f64,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            bayesian_enabled: false,
            mc_samples: 100,
            confidence_level: 0.95,
        }
    }
}

/// Result of PINN-based beamforming.
#[derive(Debug, Clone)]
pub struct PinnBeamformingResult {
    /// Beamformed image
    pub image: Array3<f32>,
    /// Uncertainty map (if enabled)
    pub uncertainty: Option<Array3<f32>>,
    /// Confidence map
    pub confidence: Option<Array3<f32>>,
    /// Inference time in seconds
    pub inference_time: f64,
    /// Training metrics (if available)
    pub metrics: TrainingMetrics,
}

/// Processing metadata.
#[derive(Debug, Clone)]
pub struct ProcessingMetadata {
    /// Processing time in seconds
    pub processing_time: f64,
    /// Model version used
    pub model_version: String,
    /// Device used
    pub device: String,
}

/// Training metrics from PINN optimization.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Total loss value
    pub total_loss: f64,
    /// Physics-based loss component
    pub physics_loss: f64,
    /// Data-fitting loss component
    pub data_loss: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Training time in seconds
    pub training_time: f64,
}

/// Model information and metadata.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Supported dimensions
    pub dimensions: Vec<usize>,
    /// Whether model is trained
    pub is_trained: bool,
}

/// Trait for PINN beamforming providers.
///
/// Implement this trait to integrate custom PINN models with the beamforming pipeline.
pub trait PinnBeamformingProvider: Send + Sync {
    /// Perform beamforming using PINN-predicted fields.
    ///
    /// # Arguments
    ///
    /// * `rf_data` - Raw RF data from sensors (shape: `[batch, channels, time]`)
    /// * `config` - Beamforming configuration
    ///
    /// # Returns
    ///
    /// Beamformed image with optional uncertainty quantification
    fn beamform(
        &self,
        rf_data: &Array3<f32>,
        config: &PinnBeamformingConfig,
    ) -> KwaversResult<PinnBeamformingResult>;

    /// Train the PINN model with training data.
    ///
    /// # Arguments
    ///
    /// * `training_data` - Pairs of (input, target) training samples
    /// * `config` - Training configuration
    ///
    /// # Returns
    ///
    /// Training metrics including loss values and convergence info
    fn train(
        &mut self,
        training_data: &[(Array3<f32>, Array3<f32>)],
        config: &PinnBeamformingConfig,
    ) -> KwaversResult<TrainingMetrics>;

    /// Estimate uncertainty in predictions.
    ///
    /// # Arguments
    ///
    /// * `rf_data` - Input RF data
    /// * `config` - Uncertainty quantification configuration
    ///
    /// # Returns
    ///
    /// Uncertainty map (variance or standard deviation)
    fn estimate_uncertainty(
        &self,
        rf_data: &Array3<f32>,
        config: &UncertaintyConfig,
    ) -> KwaversResult<Array3<f32>>;

    /// Get model information.
    fn model_info(&self) -> ModelInfo;

    /// Check if model is loaded and ready.
    fn is_ready(&self) -> bool;
}

/// Domain decomposition strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionStrategy {
    /// Spatial decomposition (partition volume)
    Spatial,
    /// Temporal decomposition (partition frames)
    Temporal,
    /// Hybrid decomposition
    Hybrid,
}

/// Load balancing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Static partitioning
    Static,
    /// Dynamic work stealing
    Dynamic,
    /// Adaptive based on GPU performance
    Adaptive,
}

/// Distributed configuration for multi-GPU inference.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of GPUs
    pub num_gpus: usize,
    /// GPU device IDs
    pub gpu_devices: Vec<usize>,
    /// Batch size per GPU
    pub batch_size_per_gpu: usize,
    /// Decomposition strategy
    pub decomposition: DecompositionStrategy,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Distributed PINN beamforming provider.
///
/// Extends the base provider with multi-GPU capabilities.
pub trait DistributedPinnProvider: PinnBeamformingProvider {
    /// Perform distributed beamforming across multiple GPUs.
    fn beamform_distributed(
        &self,
        rf_data: &Array4<f32>,
        config: &DistributedConfig,
    ) -> KwaversResult<Vec<PinnBeamformingResult>>;

    /// Get GPU utilization metrics.
    fn gpu_metrics(&self) -> Vec<GpuMetrics>;
}

/// GPU utilization metrics.
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// GPU device ID
    pub device_id: usize,
    /// Memory usage (bytes)
    pub memory_used: usize,
    /// Memory available (bytes)
    pub memory_total: usize,
    /// GPU utilization (0.0 to 1.0)
    pub utilization: f32,
    /// Temperature (Celsius)
    pub temperature: Option<f32>,
}

/// Registry for PINN beamforming providers.
///
/// Allows registering and retrieving PINN implementations at runtime.
pub struct PinnProviderRegistry {
    providers: std::collections::HashMap<String, Box<dyn PinnBeamformingProvider>>,
}

impl std::fmt::Debug for PinnProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnProviderRegistry")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl PinnProviderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            providers: std::collections::HashMap::new(),
        }
    }

    /// Register a PINN provider.
    pub fn register(&mut self, name: String, provider: Box<dyn PinnBeamformingProvider>) {
        self.providers.insert(name, provider);
    }

    /// Get a provider by name.
    pub fn get(&self, name: &str) -> Option<&dyn PinnBeamformingProvider> {
        self.providers.get(name).map(|b| &**b)
    }

    /// List available providers.
    pub fn list_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }
}

impl Default for PinnProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = PinnBeamformingConfig::default();
        assert_eq!(config.inference.batch_size, 32);
        assert!(!config.inference.use_fp16);
        assert!(!config.uncertainty.bayesian_enabled);
    }

    #[test]
    fn test_device_config() {
        let cpu = DeviceConfig::Cpu;
        assert!(matches!(cpu, DeviceConfig::Cpu));

        let gpu = DeviceConfig::Gpu { device_id: 0 };
        assert!(matches!(gpu, DeviceConfig::Gpu { .. }));
    }

    #[test]
    fn test_provider_registry() {
        let registry = PinnProviderRegistry::new();
        assert_eq!(registry.list_providers().len(), 0);
    }
}
