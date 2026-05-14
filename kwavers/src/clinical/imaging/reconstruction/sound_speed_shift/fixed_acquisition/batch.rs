//! Batch-frame reconstruction through one fixed acquisition plan.

use crate::core::error::{KwaversError, KwaversResult};

use super::super::types::SoundSpeedShiftWorkspace;
use super::super::{solve_operator_frame, solved_image_from_operator};
use super::types::{
    SoundSpeedShiftBatch, SoundSpeedShiftBatchConfig, SoundSpeedShiftBatchFrame,
    SoundSpeedShiftFrameSummary, SoundSpeedShiftObjectiveHistoryPolicy, SoundSpeedShiftPlan,
};
use super::validation::validate_frame_batch;

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
        let mut workspace = SoundSpeedShiftWorkspace::new();
        self.reconstruct_frames_with_workspace(frame_time_shifts_s, &mut workspace)
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
        validate_frame_batch(frame_time_shifts_s, self.samples.len())?;
        let mut data = vec![0.0; self.operator.rows()];
        let mut frames = Vec::with_capacity(frame_time_shifts_s.len());

        for (frame_index, time_shifts_s) in frame_time_shifts_s.iter().enumerate() {
            self.operator.rhs_from_time_shift_values_into(
                time_shifts_s,
                self.config.reference_sound_speed_m_s,
                &mut data,
            );
            solve_operator_frame(&self.operator, &data, self.config, workspace);
            let summary = objective_summary(frame_index, &workspace.objective_history)?;
            let objective_history = match batch_config.objective_history {
                SoundSpeedShiftObjectiveHistoryPolicy::Compact => Vec::new(),
                SoundSpeedShiftObjectiveHistoryPolicy::Full => workspace.objective_history.clone(),
            };
            frames.push(SoundSpeedShiftBatchFrame {
                sound_speed_shift_m_s: solved_image_from_operator(&self.operator, workspace),
                summary,
                objective_history,
            });
        }

        Ok(SoundSpeedShiftBatch {
            frames,
            rows_used: self.operator.rows(),
            rows_available: self.samples.len(),
            active_voxels: self.operator.cols(),
            model_family: self.config.model_family(),
            sampling: self.config.sampling,
            prior: self.config.prior,
        })
    }
}

fn objective_summary(
    frame_index: usize,
    history: &[f64],
) -> KwaversResult<SoundSpeedShiftFrameSummary> {
    let objective_initial = history.first().copied().ok_or_else(|| {
        KwaversError::InvalidInput("speed-shift solver produced empty objective history".to_owned())
    })?;
    let objective_final = history.last().copied().ok_or_else(|| {
        KwaversError::InvalidInput("speed-shift solver produced empty objective history".to_owned())
    })?;
    Ok(SoundSpeedShiftFrameSummary {
        frame_index,
        objective_initial,
        objective_final,
        objective_iterations: history.len().saturating_sub(1),
    })
}
