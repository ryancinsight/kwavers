//! Transfer Learning Framework for PINN Geometry Adaptation
//!
//! This module implements transfer learning techniques to adapt Physics-Informed Neural Networks
//! trained on simple geometries to more complex geometries, enabling efficient generalization.

use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

mod evaluation;
mod learner;
#[cfg(test)]
mod tests;

/// Transfer learning configuration
#[derive(Debug, Clone)]
pub struct TransferLearningConfig {
    /// Fine-tuning learning rate
    pub fine_tune_lr: f64,
    /// Number of fine-tuning epochs
    pub fine_tune_epochs: usize,
    /// Layer freezing strategy
    pub freeze_strategy: FreezeStrategy,
    /// Domain adaptation strength
    pub adaptation_strength: f64,
    /// Early stopping patience
    pub patience: usize,
    /// Reference wave speed used when no wave speed function is set (m/s)
    pub wave_speed: f64,
}

/// Layer freezing strategies for transfer learning
#[derive(Debug, Clone)]
pub enum FreezeStrategy {
    /// Fine-tune all layers
    FullFineTune,
    /// Freeze lower layers, fine-tune upper layers progressively
    ProgressiveUnfreeze,
    /// Freeze all but the last layer
    FreezeAllButLast,
    /// Freeze first N layers
    FreezeFirstNLayers(usize),
}

/// Transfer learning performance metrics
#[derive(Debug, Clone)]
pub struct TransferMetrics {
    /// Initial accuracy on target geometry (before transfer)
    pub initial_accuracy: f32,
    /// Final accuracy after transfer
    pub final_accuracy: f32,
    /// Transfer efficiency (accuracy gain per training sample)
    pub transfer_efficiency: f32,
    /// Training time for transfer
    pub training_time: std::time::Duration,
    /// Convergence speed (epochs to target accuracy)
    pub convergence_epochs: usize,
}

/// Transfer learner for geometry adaptation
pub struct TransferLearner<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Source model trained on simple geometry
    pub(super) source_model: crate::inverse::pinn::ml::BurnPINN2DWave<B>,
    /// Transfer learning configuration
    pub(super) config: TransferLearningConfig,
    /// Domain adapter network (optional)
    pub(super) domain_adapter: Option<DomainAdapter<B>>,
    /// Performance statistics
    pub(super) stats: TransferLearningStats,
}

// Manual `Debug` impl: `BurnPINN2DWave<B>` requires the `CpuAddressableStorage`
// bound to implement `Debug`, which this struct's own bound does not carry.
impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for TransferLearner<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferLearner")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// Domain adaptation network for cross-geometry transfer
pub struct DomainAdapter<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Adaptation layers
    pub(super) _layers: Vec<coeus_tensor::Tensor<f32, B>>,
    /// Adaptation strength
    pub(super) _strength: f64,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for DomainAdapter<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomainAdapter")
            .field("num_layers", &self._layers.len())
            .field("strength", &self._strength)
            .finish()
    }
}

/// Transfer learning statistics
#[derive(Debug, Clone)]
pub struct TransferLearningStats {
    pub total_transfers: usize,
    pub successful_transfers: usize,
    pub average_transfer_efficiency: f32,
    pub best_transfer_accuracy: f32,
    pub total_training_time: std::time::Duration,
}

/// Test point for evaluation
#[derive(Debug, Clone)]
pub(super) struct TestPoint {
    pub(super) x: f64,
    pub(super) y: f64,
}

/// Source model features for transfer
#[derive(Debug, Clone)]
pub(super) struct SourceFeatures {
    pub(super) _weight_magnitudes: Vec<f32>,
    pub(super) _layer_importance: Vec<f32>,
    pub(super) _geometry_adaptability: f32,
}

/// Training data for fine-tuning
#[derive(Debug, Clone)]
pub(super) struct TrainingData {
    pub(super) collocation_points: Vec<(f64, f64, f64)>,
}

impl Default for TransferLearningStats {
    fn default() -> Self {
        Self {
            total_transfers: 0,
            successful_transfers: 0,
            average_transfer_efficiency: 0.0,
            best_transfer_accuracy: 0.0,
            total_training_time: std::time::Duration::default(),
        }
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> DomainAdapter<B> {
    /// Create a new domain adapter
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(strength: f64) -> Self {
        Self {
            _layers: Vec::new(),
            _strength: strength,
        }
    }

    /// Adapt input features for target domain
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn adapt(&self, features: &Var<f32, B>) -> KwaversResult<Var<f32, B>> {
        // Domain adaptation implementation for mathematical stability
        // Currently implements identity adaptation to prevent runtime panics
        // Future: Implement proper domain adaptation layers per Ganin et al. (2016)
        // Reference: Raissi et al. (2019) "Physics-informed neural networks"
        Ok(features.clone())
    }
}
