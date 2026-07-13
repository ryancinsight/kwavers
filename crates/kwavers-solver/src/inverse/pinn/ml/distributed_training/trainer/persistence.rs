use super::super::{checkpoint::checkpoint_filename, DistributedPinnTrainer, TrainingCheckpoint};
use kwavers_core::error::{KwaversError, KwaversResult, SystemError};
use log::info;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> DistributedPinnTrainer<B> {
    /// Save checkpoint.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn save_checkpoint(&mut self) -> KwaversResult<()> {
        let checkpoint = TrainingCheckpoint {
            epoch: self.coordinator.training_state.current_epoch,
            parameters: vec![],
            optimizer_state: vec![],
            metrics: self.coordinator.training_state.global_metrics.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        let path = self
            .coordinator
            .checkpoint_manager
            .checkpoint_dir
            .join(checkpoint_filename(checkpoint.epoch));
        self.coordinator
            .checkpoint_manager
            .ensure_checkpoint_dir()?;

        let bytes = serde_json::to_vec_pretty(&checkpoint).map_err(|error| {
            KwaversError::System(SystemError::Io {
                operation: format!("serialize checkpoint {}", checkpoint.epoch),
                reason: error.to_string(),
            })
        })?;
        std::fs::write(&path, bytes)?;

        info!("Checkpoint saved: {}", path.display());
        self.coordinator.training_state.last_checkpoint = checkpoint.epoch;
        self.coordinator
            .checkpoint_manager
            .cleanup_old_checkpoints()?;

        Ok(())
    }

    /// Load checkpoint.
    /// # Errors
    /// - Returns [`crate::KwaversError::System`] if the checkpoint file is not found.
    ///
    pub fn load_checkpoint(&mut self, epoch: usize) -> KwaversResult<()> {
        let path = self
            .coordinator
            .checkpoint_manager
            .checkpoint_dir
            .join(checkpoint_filename(epoch));

        if !path.try_exists().map_err(|error| {
            KwaversError::System(SystemError::Io {
                operation: format!("check checkpoint path {}", path.display()),
                reason: error.to_string(),
            })
        })? {
            return Err(KwaversError::System(SystemError::ResourceUnavailable {
                resource: format!("Checkpoint file not found: {}", path.display()),
            }));
        }

        let bytes = std::fs::read(&path)?;
        let checkpoint: TrainingCheckpoint = serde_json::from_slice(&bytes).map_err(|error| {
            KwaversError::System(SystemError::Io {
                operation: format!("deserialize checkpoint {}", path.display()),
                reason: error.to_string(),
            })
        })?;
        self.coordinator.training_state.current_epoch = checkpoint.epoch;
        self.coordinator.training_state.last_checkpoint = checkpoint.epoch;
        self.coordinator.training_state.global_metrics = checkpoint.metrics;

        info!("Checkpoint loaded: {}", path.display());

        Ok(())
    }
}
