//! Retained fixed-acquisition batch reconstruction.

use crate::core::error::KwaversResult;
use ndarray::Array2;

use super::super::super::solved_image_from_operator_into;
use super::super::super::types::SoundSpeedShiftWorkspace;
use super::super::types::{
    SoundSpeedShiftBatch, SoundSpeedShiftBatchConfig, SoundSpeedShiftBatchFrame,
    SoundSpeedShiftBatchImageRetention, SoundSpeedShiftObjectiveHistoryPolicy, SoundSpeedShiftPlan,
    SoundSpeedShiftPlanWorkspace,
};
use super::super::validation::validate_frame_batch;
use super::solve::solve_batch_frame;
use super::summary::{objective_summary, retained_batch};

impl SoundSpeedShiftPlan {
    /// Reconstruct a batch with a temporary workspace and compact summaries.
    ///
    /// `frame_time_shifts_s[frame][row]` is indexed by original acquisition
    /// row, before sparse sampling is applied by the cached operator.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the batch is empty or
    /// any frame violates the fixed acquisition row contract.
    pub fn reconstruct_frames(
        &self,
        frame_time_shifts_s: &[&[f64]],
    ) -> KwaversResult<SoundSpeedShiftBatch> {
        let mut workspace = SoundSpeedShiftPlanWorkspace::new();
        self.reconstruct_frames_with_plan_workspace(frame_time_shifts_s, &mut workspace)
    }

    /// Reconstruct a batch with caller-owned solver scratch buffers.
    ///
    /// Full per-frame objective histories are not retained by default; use
    /// [`SoundSpeedShiftPlan::reconstruct_frames_with_options`] with
    /// [`SoundSpeedShiftObjectiveHistoryPolicy::Full`] when every objective
    /// value is needed.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the batch is empty or
    /// any frame violates the fixed acquisition row contract.
    pub fn reconstruct_frames_with_workspace(
        &self,
        frame_time_shifts_s: &[&[f64]],
        workspace: &mut SoundSpeedShiftWorkspace,
    ) -> KwaversResult<SoundSpeedShiftBatch> {
        self.reconstruct_frames_with_options(
            frame_time_shifts_s,
            SoundSpeedShiftBatchConfig::default(),
            workspace,
        )
    }

    /// Reconstruct a batch with caller-owned solver buffers and output policy.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the batch is empty or
    /// any frame violates the fixed acquisition row contract.
    pub fn reconstruct_frames_with_options(
        &self,
        frame_time_shifts_s: &[&[f64]],
        batch_config: SoundSpeedShiftBatchConfig,
        workspace: &mut SoundSpeedShiftWorkspace,
    ) -> KwaversResult<SoundSpeedShiftBatch> {
        let mut data = vec![0.0; self.operator.rows()];
        self.reconstruct_frames_core(frame_time_shifts_s, batch_config, &mut data, workspace)
    }

    /// Reconstruct a batch with caller-owned sampled-RHS and solver buffers.
    ///
    /// This path preserves the sampled-RHS allocation across repeated batch
    /// calls and applies the configured image-retention policy.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the batch is empty or
    /// any frame violates the fixed acquisition row contract.
    pub fn reconstruct_frames_with_plan_workspace(
        &self,
        frame_time_shifts_s: &[&[f64]],
        workspace: &mut SoundSpeedShiftPlanWorkspace,
    ) -> KwaversResult<SoundSpeedShiftBatch> {
        self.reconstruct_frames_with_plan_workspace_and_options(
            frame_time_shifts_s,
            SoundSpeedShiftBatchConfig::default(),
            workspace,
        )
    }

    /// Reconstruct a batch with caller-owned plan buffers and output policy.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the batch is empty or
    /// any frame violates the fixed acquisition row contract.
    pub fn reconstruct_frames_with_plan_workspace_and_options(
        &self,
        frame_time_shifts_s: &[&[f64]],
        batch_config: SoundSpeedShiftBatchConfig,
        workspace: &mut SoundSpeedShiftPlanWorkspace,
    ) -> KwaversResult<SoundSpeedShiftBatch> {
        self.reconstruct_frames_core(
            frame_time_shifts_s,
            batch_config,
            &mut workspace.sampled_rhs,
            &mut workspace.solver,
        )
    }

    fn reconstruct_frames_core(
        &self,
        frame_time_shifts_s: &[&[f64]],
        batch_config: SoundSpeedShiftBatchConfig,
        sampled_rhs: &mut Vec<f64>,
        workspace: &mut SoundSpeedShiftWorkspace,
    ) -> KwaversResult<SoundSpeedShiftBatch> {
        validate_frame_batch(frame_time_shifts_s, self.samples.len())?;
        sampled_rhs.resize(self.operator.rows(), 0.0);
        let mut output_image = match batch_config.image_retention {
            SoundSpeedShiftBatchImageRetention::Retain => Some(Array2::<f64>::zeros(self.shape)),
            SoundSpeedShiftBatchImageRetention::Discard => None,
        };
        let mut frames = Vec::with_capacity(frame_time_shifts_s.len());

        for (frame_index, time_shifts_s) in frame_time_shifts_s.iter().enumerate() {
            solve_batch_frame(self, time_shifts_s, sampled_rhs, workspace);
            let summary = objective_summary(frame_index, &workspace.objective_history)?;
            let objective_history = match batch_config.objective_history {
                SoundSpeedShiftObjectiveHistoryPolicy::Compact => Vec::new(),
                SoundSpeedShiftObjectiveHistoryPolicy::Full => workspace.objective_history.clone(),
            };
            let sound_speed_shift_m_s = output_image.as_mut().map(|image| {
                solved_image_from_operator_into(&self.operator, workspace, image);
                image.clone()
            });
            frames.push(SoundSpeedShiftBatchFrame {
                sound_speed_shift_m_s,
                summary,
                objective_history,
            });
        }

        Ok(retained_batch(self, frames))
    }
}
