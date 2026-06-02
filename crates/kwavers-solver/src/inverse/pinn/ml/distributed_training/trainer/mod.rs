mod accessors;
mod epoch;
mod persistence;

use super::{
    CheckpointManager, DistributedPinnTrainer, DistributedTrainingConfig, PerformanceStats,
    TrainingCheckpoint, TrainingCoordinator, TrainingState,
};
use kwavers_core::error::KwaversResult;
use crate::inverse::pinn::ml::{
    BurnPINN2DConfig, BurnPINN2DWave, BurnTrainingMetrics2D, BurnWave2dGeometry,
};
use burn::tensor::backend::AutodiffBackend;
use log::info;

impl<B: AutodiffBackend> DistributedPinnTrainer<B> {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn new(
        config: DistributedTrainingConfig,
        base_config: BurnPINN2DConfig,
        _geometry: BurnWave2dGeometry,
    ) -> KwaversResult<Self> {
        let decomposition =
            crate::inverse::pinn::ml::MultiGpuDecompositionStrategy::Spatial {
                dimensions: 2,
                overlap: 0.05,
            };
        let load_balancer = crate::inverse::pinn::ml::LoadBalancingAlgorithm::Dynamic {
            imbalance_threshold: 0.1,
            migration_interval: 30.0,
        };

        let multi_gpu_manager = if config.num_gpus > 1 {
            Some(
                crate::inverse::pinn::ml::MultiGpuManager::new(
                    decomposition,
                    load_balancer,
                )
                .await?,
            )
        } else {
            None
        };

        let mut model_replicas = Vec::new();

        for _gpu_id in 0..config.num_gpus {
            let device = B::Device::default();
            let model = BurnPINN2DWave::new(base_config.clone(), &device)?;
            model_replicas.push(model);
        }

        let coordinator = TrainingCoordinator {
            model_replicas,
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

    /// Train.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
}
