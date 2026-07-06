//! Distributed Training Coordinator for Multi-GPU PINN Training
//!
//! Provides distributed training coordination, gradient aggregation,
//! checkpoint management, and fault tolerance for multi-GPU PINN training.

use crate::inverse::pinn::ml::BurnTrainingMetrics2D;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

mod checkpoint;
#[cfg(test)]
mod tests;
mod trainer;

/// Gradient aggregation strategy
#[derive(Debug, Clone)]
pub enum GradientAggregation {
    Average,
    Weighted { weights: Vec<f32> },
    Adaptive { trust_threshold: f32 },
}

/// Checkpoint data for training recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub epoch: usize,
    pub parameters: Vec<f32>,
    pub optimizer_state: Vec<f32>,
    pub metrics: BurnTrainingMetrics2D,
    pub timestamp: std::time::SystemTime,
}

/// Checkpoint manager for fault tolerance
#[derive(Debug)]
pub struct CheckpointManager {
    checkpoint_dir: std::path::PathBuf,
    max_checkpoints: usize,
}

/// Training coordinator for multi-GPU PINN training
#[derive(Debug)]
pub struct TrainingCoordinator<B: AutodiffBackend> {
    model_replicas: Vec<crate::inverse::pinn::ml::BurnPINN2DWave<B>>,
    checkpoint_manager: CheckpointManager,
    training_state: TrainingState,
    performance_stats: Vec<PerformanceStats>,
}

/// Training state information
#[derive(Debug, Clone)]
pub struct TrainingState {
    pub current_epoch: usize,
    pub global_metrics: BurnTrainingMetrics2D,
    pub gpu_metrics: Vec<BurnTrainingMetrics2D>,
    pub last_checkpoint: usize,
    pub start_time: std::time::Instant,
}

/// Performance statistics per GPU
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub gpu_id: usize,
    pub epoch_time: f64,
    pub memory_usage: usize,
    pub gpu_utilization: f32,
    pub communication_time: f64,
}

/// Distributed PINN trainer
#[derive(Debug)]
pub struct DistributedPinnTrainer<B: AutodiffBackend> {
    coordinator: TrainingCoordinator<B>,
    multi_gpu_manager: Option<crate::inverse::pinn::ml::MultiGpuManager>,
    config: DistributedTrainingConfig,
}

/// Configuration for distributed training
#[derive(Debug, Clone)]
pub struct DistributedTrainingConfig {
    pub num_gpus: usize,
    pub gradient_aggregation: GradientAggregation,
    pub checkpoint_config: CheckpointConfig,
    pub communication_config: CommunicationConfig,
    pub fault_tolerance: FaultToleranceConfig,
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub directory: String,
    pub interval: usize,
    pub max_checkpoints: usize,
    pub auto_save: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            directory: "checkpoints".to_string(),
            interval: 100,
            max_checkpoints: 5,
            auto_save: true,
        }
    }
}

/// Communication configuration
#[derive(Debug, Clone)]
pub struct CommunicationConfig {
    pub backend: CommunicationBackend,
    pub compression: bool,
    pub async_comm: bool,
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            backend: CommunicationBackend::PeerToPeer,
            compression: false,
            async_comm: true,
        }
    }
}

/// Communication backend options
#[derive(Debug, Clone)]
pub enum CommunicationBackend {
    PeerToPeer,
    CpuMediated,
    Rdma,
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    pub auto_recovery: bool,
    pub max_retries: usize,
    pub graceful_degradation: bool,
    pub checkpoint_on_failure: bool,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            auto_recovery: true,
            max_retries: 3,
            graceful_degradation: true,
            checkpoint_on_failure: true,
        }
    }
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            gradient_aggregation: GradientAggregation::Average,
            checkpoint_config: CheckpointConfig::default(),
            communication_config: CommunicationConfig::default(),
            fault_tolerance: FaultToleranceConfig::default(),
        }
    }
}
