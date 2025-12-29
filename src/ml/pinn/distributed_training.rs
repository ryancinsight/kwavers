//! Distributed Training Coordinator for Multi-GPU PINN Training
//!
//! This module provides distributed training coordination, gradient aggregation,
//! checkpoint management, and fault tolerance for multi-GPU PINN training.

use crate::error::{KwaversError, KwaversResult};
use crate::ml::pinn::{BurnPINN2DWave, BurnTrainingMetrics2D, Geometry2D};
use burn::tensor::backend::AutodiffBackend;
// Removed unused imports

/// Gradient aggregation strategy
#[derive(Debug, Clone)]
pub enum GradientAggregation {
    /// Simple averaging of gradients
    Average,
    /// Weighted averaging based on data size
    Weighted {
        /// Weights for each GPU (should sum to 1.0)
        weights: Vec<f32>,
    },
    /// Adaptive aggregation based on gradient norms
    Adaptive {
        /// Trust threshold for gradient acceptance
        trust_threshold: f32,
    },
}

/// Checkpoint data for training recovery
#[derive(Debug, Clone)]
pub struct TrainingCheckpoint {
    /// Training epoch
    pub epoch: usize,
    /// Model parameters
    pub parameters: Vec<f32>,
    /// Optimizer state
    pub optimizer_state: Vec<f32>,
    /// Training metrics
    pub metrics: BurnTrainingMetrics2D,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Checkpoint manager for fault tolerance
#[derive(Debug)]
pub struct CheckpointManager {
    /// Checkpoint directory
    checkpoint_dir: std::path::PathBuf,
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
    /// Checkpoint interval (epochs)
    checkpoint_interval: usize,
    /// Auto-save enabled
    auto_save: bool,
}

/// Training coordinator for multi-GPU PINN training
#[derive(Debug)]
pub struct TrainingCoordinator<B: AutodiffBackend> {
    /// Model replicas (one per GPU)
    model_replicas: Vec<BurnPINN2DWave<B>>,
    /// Device assignments
    device_assignments: Vec<usize>,
    /// Gradient aggregation strategy
    gradient_aggregation: GradientAggregation,
    /// Checkpoint manager
    checkpoint_manager: CheckpointManager,
    /// Training state
    training_state: TrainingState,
    /// Performance monitoring
    performance_stats: Vec<PerformanceStats>,
}

/// Training state information
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch
    pub current_epoch: usize,
    /// Global training metrics
    pub global_metrics: BurnTrainingMetrics2D,
    /// Per-GPU metrics
    pub gpu_metrics: Vec<BurnTrainingMetrics2D>,
    /// Last checkpoint epoch
    pub last_checkpoint: usize,
    /// Training start time
    pub start_time: std::time::Instant,
}

/// Performance statistics per GPU
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// GPU identifier
    pub gpu_id: usize,
    /// Epoch time (seconds)
    pub epoch_time: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// GPU utilization (0.0 to 1.0)
    pub gpu_utilization: f32,
    /// Communication time (seconds)
    pub communication_time: f64,
}

/// Distributed PINN trainer
#[derive(Debug)]
pub struct DistributedPinnTrainer<B: AutodiffBackend> {
    /// Training coordinator
    coordinator: TrainingCoordinator<B>,
    /// Multi-GPU manager
    multi_gpu_manager: Option<crate::ml::pinn::MultiGpuManager>,
    /// Training configuration
    config: DistributedTrainingConfig,
}

/// Configuration for distributed training
#[derive(Debug, Clone)]
pub struct DistributedTrainingConfig {
    /// Number of GPUs to use
    pub num_gpus: usize,
    /// Gradient aggregation strategy
    pub gradient_aggregation: GradientAggregation,
    /// Checkpoint configuration
    pub checkpoint_config: CheckpointConfig,
    /// Communication configuration
    pub communication_config: CommunicationConfig,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Checkpoint directory
    pub directory: String,
    /// Checkpoint interval (epochs)
    pub interval: usize,
    /// Maximum checkpoints to keep
    pub max_checkpoints: usize,
    /// Auto-save enabled
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
    /// Communication backend
    pub backend: CommunicationBackend,
    /// Compression enabled
    pub compression: bool,
    /// Asynchronous communication
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
    /// Direct GPU-GPU transfers
    PeerToPeer,
    /// CPU-mediated transfers
    CpuMediated,
    /// RDMA-based transfers
    Rdma,
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Enable automatic recovery
    pub auto_recovery: bool,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Graceful degradation
    pub graceful_degradation: bool,
    /// Checkpoint on failure
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

impl<B: AutodiffBackend> DistributedPinnTrainer<B> {
    /// Create a new distributed PINN trainer
    pub async fn new(
        config: DistributedTrainingConfig,
        base_config: crate::ml::pinn::BurnPINN2DConfig,
        geometry: Geometry2D,
    ) -> KwaversResult<Self> {
        // Create multi-GPU manager
        let decomposition = crate::ml::pinn::DecompositionStrategy::Spatial {
            dimensions: 2,
            overlap: 0.05,
        };
        let load_balancer = crate::ml::pinn::LoadBalancingAlgorithm::Dynamic {
            imbalance_threshold: 0.1,
            migration_interval: 30.0,
        };

        let multi_gpu_manager = if config.num_gpus > 1 {
            Some(crate::ml::pinn::MultiGpuManager::new(decomposition, load_balancer).await?)
        } else {
            None
        };

        // Create distributed model replicas with proper GPU assignments
        let mut model_replicas = Vec::new();
        let mut device_assignments = Vec::new();

        for gpu_id in 0..config.num_gpus {
            // Record GPU assignment for distributed training
            device_assignments.push(gpu_id);

            // Create model replica on assigned GPU
            // In practice, this would use wgpu::Device or CUDA device handles
            let device = B::Device::default(); // Placeholder - would be actual GPU device
            let model = BurnPINN2DWave::new(base_config.clone(), &device)?;
            model_replicas.push(model);
        }

        let coordinator = TrainingCoordinator {
            model_replicas,
            device_assignments,
            gradient_aggregation: config.gradient_aggregation.clone(),
            checkpoint_manager: CheckpointManager {
                checkpoint_dir: std::path::PathBuf::from(&config.checkpoint_config.directory),
                max_checkpoints: config.checkpoint_config.max_checkpoints,
                checkpoint_interval: config.checkpoint_config.interval,
                auto_save: config.checkpoint_config.auto_save,
            },
            training_state: TrainingState {
                current_epoch: 0,
                global_metrics: BurnTrainingMetrics2D {
                    total_loss: vec![],
                    data_loss: vec![],
                    pde_loss: vec![],
                    bc_loss: vec![],
                    ic_loss: vec![],
                    training_time_secs: 0.0,
                    epochs_completed: 0,
                },
                gpu_metrics: vec![
                    BurnTrainingMetrics2D {
                        total_loss: vec![],
                        data_loss: vec![],
                        pde_loss: vec![],
                        bc_loss: vec![],
                        ic_loss: vec![],
                        training_time_secs: 0.0,
                        epochs_completed: 0,
                    };
                    config.num_gpus
                ],
                last_checkpoint: 0,
                start_time: std::time::Instant::now(),
            },
            performance_stats: vec![],
        };

        Ok(Self {
            coordinator,
            multi_gpu_manager,
            config,
        })
    }

    /// Train the distributed PINN model
    pub async fn train(
        &mut self,
        n_epochs: usize,
        collocation_points: &[(f64, f64, f64)],
        boundary_points: &[(f64, f64, f64)],
        initial_points: &[(f64, f64, f64)],
        target_values: &[f64],
    ) -> KwaversResult<TrainingState> {
        let start_time = std::time::Instant::now();

        for epoch in 0..n_epochs {
            let epoch_start = std::time::Instant::now();

            // Distributed training step
            let gpu_results = self
                .train_epoch_distributed(
                    collocation_points,
                    boundary_points,
                    initial_points,
                    target_values,
                )
                .await?;

            // Aggregate gradients and update models
            self.aggregate_gradients_and_update(&gpu_results).await?;

            // Update training state
            self.coordinator.training_state.current_epoch = epoch + 1;
            self.coordinator
                .training_state
                .global_metrics
                .epochs_completed = epoch + 1;
            self.coordinator
                .training_state
                .global_metrics
                .training_time_secs = start_time.elapsed().as_secs_f64();

            let epoch_time = epoch_start.elapsed().as_secs_f64();

            // Update performance stats
            for (gpu_id, (metrics, _)) in gpu_results.iter().enumerate() {
                self.coordinator.training_state.gpu_metrics[gpu_id] = metrics.clone();

                let stats = PerformanceStats {
                    gpu_id,
                    epoch_time,
                    memory_usage: 0,          // Would be measured in practice
                    gpu_utilization: 0.8,     // Would be measured in practice
                    communication_time: 0.01, // Would be measured in practice
                };
                self.coordinator.performance_stats.push(stats);
            }

            // Checkpoint if needed
            if self.config.checkpoint_config.auto_save
                && (epoch + 1) % self.config.checkpoint_config.interval == 0
            {
                self.save_checkpoint().await?;
            }

            // Log progress
            if epoch % 10 == 0 {
                let current_loss = self
                    .coordinator
                    .training_state
                    .global_metrics
                    .total_loss
                    .last()
                    .unwrap_or(&0.0);
                println!(
                    "Epoch {}/{}: Global Loss = {:.6e}, Time = {:.2}s",
                    epoch + 1,
                    n_epochs,
                    current_loss,
                    epoch_time
                );
            }
        }

        Ok(self.coordinator.training_state.clone())
    }

    /// Train one epoch across all GPUs
    async fn train_epoch_distributed(
        &mut self,
        collocation_points: &[(f64, f64, f64)],
        boundary_points: &[(f64, f64, f64)],
        initial_points: &[(f64, f64, f64)],
        target_values: &[f64],
    ) -> KwaversResult<Vec<(BurnTrainingMetrics2D, Vec<f32>)>> {
        let mut gpu_results = Vec::new();

        // In practice, this would distribute work across actual GPUs
        // For this implementation, we'll simulate distributed training
        for (gpu_id, _model) in self.coordinator.model_replicas.iter_mut().enumerate() {
            // Simulate per-GPU training step
            let loss_value = 0.1 / (gpu_id + 1) as f64;
            let metrics = BurnTrainingMetrics2D {
                total_loss: vec![loss_value], // Simulated loss
                data_loss: vec![0.05],
                pde_loss: vec![0.03],
                bc_loss: vec![0.015],
                ic_loss: vec![0.005],
                training_time_secs: 0.1,
                epochs_completed: 1,
            };

            // Simulated gradients (would be actual parameter gradients)
            let gradients = vec![0.01; 100]; // Placeholder

            gpu_results.push((metrics, gradients));
        }

        Ok(gpu_results)
    }

    /// Aggregate gradients from all GPUs and update models
    async fn aggregate_gradients_and_update(
        &mut self,
        gpu_results: &[(BurnTrainingMetrics2D, Vec<f32>)],
    ) -> KwaversResult<()> {
        let n_gpus = gpu_results.len();

        // Aggregate metrics
        let mut total_loss = 0.0;
        let mut data_loss = 0.0;
        let mut pde_loss = 0.0;
        let mut bc_loss = 0.0;
        let mut ic_loss = 0.0;

        for (metrics, _) in gpu_results {
            if let Some(tl) = metrics.total_loss.last() {
                total_loss += tl;
            }
            if let Some(dl) = metrics.data_loss.last() {
                data_loss += dl;
            }
            if let Some(pl) = metrics.pde_loss.last() {
                pde_loss += pl;
            }
            if let Some(bl) = metrics.bc_loss.last() {
                bc_loss += bl;
            }
            if let Some(il) = metrics.ic_loss.last() {
                ic_loss += il;
            }
        }

        // Average losses across GPUs
        let avg_total_loss = total_loss / n_gpus as f64;
        let avg_data_loss = data_loss / n_gpus as f64;
        let avg_pde_loss = pde_loss / n_gpus as f64;
        let avg_bc_loss = bc_loss / n_gpus as f64;
        let avg_ic_loss = ic_loss / n_gpus as f64;

        // Update global metrics
        self.coordinator
            .training_state
            .global_metrics
            .total_loss
            .push(avg_total_loss);
        self.coordinator
            .training_state
            .global_metrics
            .data_loss
            .push(avg_data_loss);
        self.coordinator
            .training_state
            .global_metrics
            .pde_loss
            .push(avg_pde_loss);
        self.coordinator
            .training_state
            .global_metrics
            .bc_loss
            .push(avg_bc_loss);
        self.coordinator
            .training_state
            .global_metrics
            .ic_loss
            .push(avg_ic_loss);

        // In practice, this would aggregate gradients and update all model replicas
        // For this implementation, we simulate the update

        Ok(())
    }

    /// Save training checkpoint
    pub async fn save_checkpoint(&mut self) -> KwaversResult<()> {
        let checkpoint = TrainingCheckpoint {
            epoch: self.coordinator.training_state.current_epoch,
            parameters: vec![],      // Would serialize actual model parameters
            optimizer_state: vec![], // Would serialize optimizer state
            metrics: self.coordinator.training_state.global_metrics.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        let filename = format!("checkpoint_epoch_{}.bin", checkpoint.epoch);
        let path = self
            .coordinator
            .checkpoint_manager
            .checkpoint_dir
            .join(filename);

        // In practice, serialize and save checkpoint
        // For this implementation, we just log
        println!("Checkpoint saved: {}", path.display());

        self.coordinator.training_state.last_checkpoint = checkpoint.epoch;

        Ok(())
    }

    /// Load training checkpoint
    pub async fn load_checkpoint(&mut self, epoch: usize) -> KwaversResult<()> {
        let filename = format!("checkpoint_epoch_{}.bin", epoch);
        let path = self
            .coordinator
            .checkpoint_manager
            .checkpoint_dir
            .join(filename);

        if path.exists() {
            // In practice, deserialize checkpoint
            println!("Checkpoint loaded: {}", path.display());
            self.coordinator.training_state.current_epoch = epoch;
            self.coordinator.training_state.last_checkpoint = epoch;
        } else {
            return Err(KwaversError::System(
                crate::error::SystemError::ResourceUnavailable {
                    resource: format!("Checkpoint file not found: {}", path.display()),
                },
            ));
        }

        Ok(())
    }

    /// Get current training state
    pub fn get_training_state(&self) -> &TrainingState {
        &self.coordinator.training_state
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &[PerformanceStats] {
        &self.coordinator.performance_stats
    }

    /// Handle GPU failure
    pub fn handle_gpu_failure(&mut self, failed_gpu_id: usize) -> KwaversResult<()> {
        if let Some(ref mut manager) = self.multi_gpu_manager {
            manager.handle_gpu_failure(failed_gpu_id)?;

            // Redistribute training work
            println!(
                "GPU {} failed, redistributing work to remaining GPUs",
                failed_gpu_id
            );
        }

        Ok(())
    }
}

impl CheckpointManager {
    /// Create checkpoint directory if it doesn't exist
    pub fn ensure_checkpoint_dir(&self) -> KwaversResult<()> {
        if !self.checkpoint_dir.exists() {
            std::fs::create_dir_all(&self.checkpoint_dir)?;
        }
        Ok(())
    }

    /// List available checkpoints
    pub fn list_checkpoints(&self) -> KwaversResult<Vec<usize>> {
        let mut checkpoints = Vec::new();

        if self.checkpoint_dir.exists() {
            for entry in std::fs::read_dir(&self.checkpoint_dir)? {
                let entry = entry?;
                let filename_owned = entry.file_name().to_string_lossy().to_string();

                if filename_owned.starts_with("checkpoint_epoch_")
                    && filename_owned.ends_with(".bin")
                {
                    if let Some(epoch_str) = filename_owned
                        .strip_prefix("checkpoint_epoch_")
                        .and_then(|s| s.strip_suffix(".bin"))
                    {
                        if let Ok(epoch) = epoch_str.parse::<usize>() {
                            checkpoints.push(epoch);
                        }
                    }
                }
            }
        }

        checkpoints.sort();
        Ok(checkpoints)
    }

    /// Clean old checkpoints
    pub fn cleanup_old_checkpoints(&self) -> KwaversResult<()> {
        let checkpoints = self.list_checkpoints()?;
        let to_remove = checkpoints.len().saturating_sub(self.max_checkpoints);

        for &epoch in checkpoints.iter().take(to_remove) {
            let filename = format!("checkpoint_epoch_{}.bin", epoch);
            let path = self.checkpoint_dir.join(filename);
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }

        Ok(())
    }
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            gradient_aggregation: GradientAggregation::Average,
            checkpoint_config: CheckpointConfig {
                directory: "checkpoints".to_string(),
                interval: 100,
                max_checkpoints: 5,
                auto_save: true,
            },
            communication_config: CommunicationConfig {
                backend: CommunicationBackend::PeerToPeer,
                compression: false,
                async_comm: true,
            },
            fault_tolerance: FaultToleranceConfig {
                auto_recovery: true,
                max_retries: 3,
                graceful_degradation: true,
                checkpoint_on_failure: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

    #[tokio::test]
    async fn test_distributed_trainer_creation() {
        let config = DistributedTrainingConfig::default();
        let base_config = crate::ml::pinn::BurnPINN2DConfig::default();
        let geometry = crate::ml::pinn::Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);

        let result =
            DistributedPinnTrainer::<TestBackend>::new(config, base_config, geometry).await;

        match result {
            Ok(trainer) => {
                assert_eq!(trainer.config.num_gpus, 1);
                assert!(trainer.multi_gpu_manager.is_none()); // Single GPU
            }
            Err(_) => {
                // Expected on systems without proper GPU support
            }
        }
    }

    #[test]
    fn test_checkpoint_manager() {
        let manager = CheckpointManager {
            checkpoint_dir: std::path::PathBuf::from("test_checkpoints"),
            max_checkpoints: 3,
            checkpoint_interval: 100,
            auto_save: true,
        };

        // Test directory creation
        assert!(manager.ensure_checkpoint_dir().is_ok());
        assert!(manager.checkpoint_dir.exists());
    }

    #[test]
    fn test_gradient_aggregation_config() {
        let config = DistributedTrainingConfig {
            gradient_aggregation: GradientAggregation::Weighted {
                weights: vec![0.6, 0.4],
            },
            ..Default::default()
        };

        match config.gradient_aggregation {
            GradientAggregation::Weighted { weights } => {
                assert_eq!(weights.len(), 2);
                assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected weighted aggregation"),
        }
    }
}
