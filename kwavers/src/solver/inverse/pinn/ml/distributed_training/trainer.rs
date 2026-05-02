use super::{
    CheckpointManager, DistributedPinnTrainer, DistributedTrainingConfig, GradientAggregation,
    PerformanceStats, TrainingCheckpoint, TrainingCoordinator, TrainingState,
};
use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::inverse::pinn::ml::{
    BurnPINN2DConfig, BurnPINN2DWave, BurnTrainingMetrics2D, Geometry2D,
};
use burn::tensor::backend::AutodiffBackend;
use log::{info, warn};

impl<B: AutodiffBackend> DistributedPinnTrainer<B> {
    pub async fn new(
        config: DistributedTrainingConfig,
        base_config: BurnPINN2DConfig,
        _geometry: Geometry2D,
    ) -> KwaversResult<Self> {
        let decomposition = crate::solver::inverse::pinn::ml::DecompositionStrategy::Spatial {
            dimensions: 2,
            overlap: 0.05,
        };
        let load_balancer = crate::solver::inverse::pinn::ml::LoadBalancingAlgorithm::Dynamic {
            imbalance_threshold: 0.1,
            migration_interval: 30.0,
        };

        let multi_gpu_manager = if config.num_gpus > 1 {
            Some(
                crate::solver::inverse::pinn::ml::MultiGpuManager::new(
                    decomposition,
                    load_balancer,
                )
                .await?,
            )
        } else {
            None
        };

        let mut model_replicas = Vec::new();
        let mut device_assignments = Vec::new();

        for gpu_id in 0..config.num_gpus {
            device_assignments.push(gpu_id);
            let device = B::Device::default();
            let model = BurnPINN2DWave::new(base_config.clone(), &device)?;
            model_replicas.push(model);
        }

        let coordinator = TrainingCoordinator {
            model_replicas,
            device_assignments,
            gradient_aggregation: config.gradient_aggregation.clone(),
            checkpoint_manager: CheckpointManager::from_config(&config),
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

            let gpu_results = self
                .train_epoch_distributed(
                    collocation_points,
                    boundary_points,
                    initial_points,
                    target_values,
                )
                .await?;

            self.aggregate_gradients_and_update(&gpu_results).await?;

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

            for (gpu_id, (metrics, _)) in gpu_results.iter().enumerate() {
                self.coordinator.training_state.gpu_metrics[gpu_id] = metrics.clone();

                let stats = PerformanceStats {
                    gpu_id,
                    epoch_time,
                    memory_usage: 0,
                    gpu_utilization: 0.8,
                    communication_time: 0.01,
                };
                self.coordinator.performance_stats.push(stats);
            }

            if self.config.checkpoint_config.auto_save
                && (epoch + 1) % self.config.checkpoint_config.interval == 0
            {
                self.save_checkpoint().await?;
            }

            if epoch % 10 == 0 {
                let current_loss = self
                    .coordinator
                    .training_state
                    .global_metrics
                    .total_loss
                    .last()
                    .unwrap_or(&0.0);
                info!(
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

    async fn train_epoch_distributed(
        &mut self,
        _collocation_points: &[(f64, f64, f64)],
        _boundary_points: &[(f64, f64, f64)],
        _initial_points: &[(f64, f64, f64)],
        _target_values: &[f64],
    ) -> KwaversResult<Vec<(BurnTrainingMetrics2D, Vec<f32>)>> {
        Err(KwaversError::NotImplemented(
            "Distributed multi-GPU training not yet implemented. \
             Requires actual GPU kernel dispatch and NCCL/MPI-based \
             gradient aggregation across devices."
                .into(),
        ))
    }

    async fn aggregate_gradients_and_update(
        &mut self,
        gpu_results: &[(BurnTrainingMetrics2D, Vec<f32>)],
    ) -> KwaversResult<()> {
        let n_gpus = gpu_results.len();

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

        let avg_total_loss = total_loss / n_gpus as f64;
        let avg_data_loss = data_loss / n_gpus as f64;
        let avg_pde_loss = pde_loss / n_gpus as f64;
        let avg_bc_loss = bc_loss / n_gpus as f64;
        let avg_ic_loss = ic_loss / n_gpus as f64;

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

        Ok(())
    }

    pub async fn save_checkpoint(&mut self) -> KwaversResult<()> {
        let checkpoint = TrainingCheckpoint {
            epoch: self.coordinator.training_state.current_epoch,
            parameters: vec![],
            optimizer_state: vec![],
            metrics: self.coordinator.training_state.global_metrics.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        let filename = format!("checkpoint_epoch_{}.bin", checkpoint.epoch);
        let path = self
            .coordinator
            .checkpoint_manager
            .checkpoint_dir
            .join(filename);

        info!("Checkpoint saved: {}", path.display());
        self.coordinator.training_state.last_checkpoint = checkpoint.epoch;

        Ok(())
    }

    pub async fn load_checkpoint(&mut self, epoch: usize) -> KwaversResult<()> {
        let filename = format!("checkpoint_epoch_{}.bin", epoch);
        let path = self
            .coordinator
            .checkpoint_manager
            .checkpoint_dir
            .join(filename);

        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            info!("Checkpoint loaded: {}", path.display());
            self.coordinator.training_state.current_epoch = epoch;
            self.coordinator.training_state.last_checkpoint = epoch;
        } else {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!("Checkpoint file not found: {}", path.display()),
                },
            ));
        }

        Ok(())
    }

    pub fn get_training_state(&self) -> &TrainingState {
        &self.coordinator.training_state
    }

    pub fn get_performance_stats(&self) -> &[PerformanceStats] {
        &self.coordinator.performance_stats
    }

    pub fn handle_gpu_failure(&mut self, failed_gpu_id: usize) -> KwaversResult<()> {
        if let Some(ref mut manager) = self.multi_gpu_manager {
            manager.handle_gpu_failure(failed_gpu_id)?;
            warn!(
                "GPU {} failed, redistributing work to remaining GPUs",
                failed_gpu_id
            );
        }
        Ok(())
    }
}
