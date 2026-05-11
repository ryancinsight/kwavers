use super::super::{DistributedPinnTrainer, PerformanceStats, TrainingState};
use crate::core::error::KwaversResult;
use burn::tensor::backend::AutodiffBackend;
use log::warn;

impl<B: AutodiffBackend> DistributedPinnTrainer<B> {
    /// Get training state.
    pub fn get_training_state(&self) -> &TrainingState {
        &self.coordinator.training_state
    }

    /// Get performance stats.
    pub fn get_performance_stats(&self) -> &[PerformanceStats] {
        &self.coordinator.performance_stats
    }

    /// Handle gpu failure.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
