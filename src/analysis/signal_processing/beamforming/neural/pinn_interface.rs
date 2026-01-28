//! PINN beamforming interface abstraction.
//!
//! This module provides trait-based abstractions for physics-informed neural network
//! beamforming, decoupling the analysis layer from solver implementation details.
//!
//! ## Architecture
//!
//! The interface follows the Dependency Inversion Principle:
//! - Analysis layer depends on traits (abstractions)
//! - Solver layer implements traits (concrete implementations)
//! - No direct dependency from analysis → solver
//!
//! ## Layer Separation
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │  Analysis Layer (Layer 7)                   │
//! │  - Neural beamforming algorithms            │
//! │  - Depends on: PinnBeamformingProvider      │
//! └─────────────────────────────────────────────┘
//!                     ↑ (trait)
//!                     │
//! ┌─────────────────────────────────────────────┐
//! │  Solver Layer (Layer 4)                     │
//! │  - PINN implementations                      │
//! │  - Implements: PinnBeamformingProvider      │
//! └─────────────────────────────────────────────┘
//! ```

use crate::core::error::KwaversResult;
use ndarray::{Array3, Array4};
use serde::{Deserialize, Serialize};

/// Configuration for PINN-based beamforming (solver-agnostic).
///
/// This configuration type belongs to the analysis layer and contains
/// only the parameters needed for beamforming, not solver-specific details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnBeamformingConfig {
    /// Model inference configuration
    pub inference: InferenceConfig,
    /// Uncertainty quantification settings
    pub uncertainty: UncertaintyConfig,
    /// Physics constraint weights
    pub physics_weights: PhysicsWeights,
    /// Device configuration (CPU/GPU)
    pub device: DeviceConfig,
}

impl Default for PinnBeamformingConfig {
    fn default() -> Self {
        Self {
            inference: InferenceConfig::default(),
            uncertainty: UncertaintyConfig::default(),
            physics_weights: PhysicsWeights::default(),
            device: DeviceConfig::default(),
        }
    }
}

/// Inference configuration for PINN models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference
    pub batch_size: usize,
    /// Use mixed precision (FP16)
    pub use_fp16: bool,
    /// Enable JIT compilation
    pub jit_enabled: bool,
    /// Model cache path
    pub model_cache_path: Option<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            use_fp16: false,
            jit_enabled: false,
            model_cache_path: None,
        }
    }
}

/// Uncertainty quantification configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyConfig {
    /// Enable Bayesian uncertainty estimation
    pub bayesian_enabled: bool,
    /// Number of Monte Carlo samples
    pub mc_samples: usize,
    /// Dropout rate for uncertainty estimation
    pub dropout_rate: f32,
    /// Confidence interval (e.g., 0.95 for 95%)
    pub confidence_level: f32,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            bayesian_enabled: false,
            mc_samples: 100,
            dropout_rate: 0.1,
            confidence_level: 0.95,
        }
    }
}

/// Physics constraint weights for PINN training/inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsWeights {
    /// Wave equation residual weight
    pub wave_equation: f32,
    /// Boundary condition weight
    pub boundary_condition: f32,
    /// Reciprocity constraint weight
    pub reciprocity: f32,
    /// Coherence constraint weight
    pub coherence: f32,
}

impl Default for PhysicsWeights {
    fn default() -> Self {
        Self {
            wave_equation: 1.0,
            boundary_condition: 1.0,
            reciprocity: 0.1,
            coherence: 0.1,
        }
    }
}

/// Device configuration for computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    /// CPU computation
    Cpu,
    /// Single GPU
    Gpu { device_id: usize },
    /// Multiple GPUs
    MultiGpu {
        device_ids: Vec<usize>,
        strategy: DistributionStrategy,
    },
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Strategy for distributing computation across GPUs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistributionStrategy {
    /// Split data spatially
    DataParallel,
    /// Split model layers
    ModelParallel,
    /// Hybrid data + model parallelism
    Hybrid,
}

/// Training metrics from PINN model (solver-agnostic).
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Total loss value
    pub total_loss: f32,
    /// Physics loss (PDE residual)
    pub physics_loss: f32,
    /// Data loss (fitting error)
    pub data_loss: f32,
    /// Training iterations completed
    pub iterations: usize,
    /// Training time (seconds)
    pub training_time: f64,
}

/// Result from PINN beamforming inference.
#[derive(Debug, Clone)]
pub struct PinnBeamformingResult {
    /// Beamformed image
    pub image: Array3<f32>,
    /// Uncertainty map (optional)
    pub uncertainty: Option<Array3<f32>>,
    /// Confidence scores per pixel
    pub confidence: Option<Array3<f32>>,
    /// Inference time (seconds)
    pub inference_time: f64,
    /// Model metrics
    pub metrics: TrainingMetrics,
}

/// Distributed PINN processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// GPU device IDs
    pub gpu_devices: Vec<usize>,
    /// Decomposition strategy
    pub decomposition: DecompositionStrategy,
    /// Load balancing algorithm
    pub load_balancing: LoadBalancingStrategy,
    /// Enable fault tolerance
    pub fault_tolerance: bool,
    /// Communication backend
    pub communication: CommunicationBackend,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            gpu_devices: vec![0],
            decomposition: DecompositionStrategy::Spatial,
            load_balancing: LoadBalancingStrategy::Static,
            fault_tolerance: false,
            communication: CommunicationBackend::Shared,
        }
    }
}

/// Strategy for decomposing workload.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Spatial domain decomposition
    Spatial,
    /// Temporal decomposition
    Temporal,
    /// Hybrid spatial-temporal
    Hybrid,
}

/// Load balancing strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Static assignment
    Static,
    /// Dynamic work stealing
    Dynamic,
    /// Prediction-based
    Predictive,
}

/// Communication backend for multi-GPU.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CommunicationBackend {
    /// Shared memory (single node)
    Shared,
    /// NCCL (NVIDIA)
    Nccl,
    /// Gloo (cross-platform)
    Gloo,
}

/// Provider trait for PINN-based beamforming.
///
/// This trait abstracts the solver implementation details, allowing the
/// analysis layer to work with any PINN implementation that satisfies
/// this interface.
///
/// ## Implementation Notes
///
/// Implementations should be in the solver layer and registered with
/// the analysis layer through dependency injection or feature flags.
pub trait PinnBeamformingProvider: Send + Sync {
    /// Perform beamforming inference on RF data.
    ///
    /// # Arguments
    ///
    /// * `rf_data` - Raw RF channel data (channels × samples × frames)
    /// * `config` - PINN beamforming configuration
    ///
    /// # Returns
    ///
    /// Beamformed image with optional uncertainty quantification
    fn beamform(
        &self,
        rf_data: &Array3<f32>,
        config: &PinnBeamformingConfig,
    ) -> KwaversResult<PinnBeamformingResult>;

    /// Train or fine-tune the PINN model.
    ///
    /// # Arguments
    ///
    /// * `training_data` - Training dataset (inputs × outputs)
    /// * `config` - Training configuration
    ///
    /// # Returns
    ///
    /// Training metrics
    fn train(
        &mut self,
        training_data: &[(Array3<f32>, Array3<f32>)],
        config: &PinnBeamformingConfig,
    ) -> KwaversResult<TrainingMetrics>;

    /// Compute uncertainty estimates.
    ///
    /// # Arguments
    ///
    /// * `rf_data` - RF channel data
    /// * `config` - Uncertainty configuration
    ///
    /// # Returns
    ///
    /// Uncertainty map
    fn estimate_uncertainty(
        &self,
        rf_data: &Array3<f32>,
        config: &UncertaintyConfig,
    ) -> KwaversResult<Array3<f32>>;

    /// Get model information.
    fn model_info(&self) -> ModelInfo;
}

/// Information about the PINN model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Supported dimensions (1D, 2D, 3D)
    pub dimensions: Vec<usize>,
    /// Is the model trained?
    pub is_trained: bool,
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
        let mut registry = PinnProviderRegistry::new();
        assert_eq!(registry.list_providers().len(), 0);
    }
}
