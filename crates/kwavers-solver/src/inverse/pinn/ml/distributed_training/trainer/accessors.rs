use super::super::{DistributedPinnTrainer, PerformanceStats, TrainingState};
use kwavers_core::error::KwaversResult;
use log::warn;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> DistributedPinnTrainer<B> {
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
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
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
