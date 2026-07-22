//! Checkpoint serialisation for `BeamformingTrainer`.
//!
//! Serialises training configuration, physics loss weights, and per-epoch
//! metrics to a JSON file. When a Burn model is available, model weights
//! should be saved alongside this metadata.

use super::BeamformingTrainer;
use kwavers_core::error::{KwaversError, KwaversResult};
use log::debug;

impl BeamformingTrainer {
    /// Save model checkpoint with training state.
    ///
    /// Serialises the training configuration, physics loss weights, and
    /// per-epoch metrics into a JSON file.  When a Burn model is available,
    /// model weights should be saved alongside this metadata.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub(super) fn save_checkpoint(&self, epoch: usize) -> KwaversResult<()> {
        let checkpoint_dir = &self.config.checkpoint_dir;

        std::fs::create_dir_all(checkpoint_dir).map_err(|e| {
            KwaversError::InternalError(format!("Failed to create checkpoint dir: {e}"))
        })?;

        let checkpoint_path = format!("{}/checkpoint_epoch_{:04}.json", checkpoint_dir, epoch);

        let state = format!(
            concat!(
                "{{\n",
                "  \"epoch\": {},\n",
                "  \"learning_rate\": {},\n",
                "  \"lambda_data\": {},\n",
                "  \"lambda_physics\": {},\n",
                "  \"best_val_loss\": {},\n",
                "  \"best_epoch\": {},\n",
                "  \"total_epochs_trained\": {},\n",
                "  \"physics_loss_weights\": {{\n",
                "    \"reciprocity\": {},\n",
                "    \"coherence\": {},\n",
                "    \"sparsity\": {}\n",
                "  }}\n",
                "}}"
            ),
            epoch,
            self.config.learning_rate,
            self.config.lambda_data,
            self.config.lambda_physics,
            self.history.best_val_loss,
            self.history.best_epoch,
            self.history.epochs.len(),
            self.physics_loss.reciprocity_weight,
            self.physics_loss.coherence_weight,
            self.physics_loss.sparsity_weight,
        );

        std::fs::write(&checkpoint_path, state)
            .map_err(|e| KwaversError::InternalError(format!("Failed to write checkpoint: {e}")))?;

        if self.config.verbose {
            debug!("Saved checkpoint: {checkpoint_path}");
        }

        Ok(())
    }
}