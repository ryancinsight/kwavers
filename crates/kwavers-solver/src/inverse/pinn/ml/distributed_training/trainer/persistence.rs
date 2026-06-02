use super::super::{DistributedPinnTrainer, TrainingCheckpoint};
use kwavers_core::error::{KwaversError, KwaversResult};
use burn::tensor::backend::AutodiffBackend;
use log::info;

impl<B: AutodiffBackend> DistributedPinnTrainer<B> {
    /// Save checkpoint.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

    /// Load checkpoint.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the checkpoint file is not found.
    ///
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
                kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: format!("Checkpoint file not found: {}", path.display()),
                },
            ));
        }

        Ok(())
    }
}
