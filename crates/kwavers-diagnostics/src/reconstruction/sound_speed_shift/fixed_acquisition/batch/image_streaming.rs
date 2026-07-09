//! Borrowed-image streaming for fixed-acquisition batch reconstruction.

use kwavers_core::error::KwaversResult;
use leto::Array2;

use super::super::super::solved_image_from_operator_into;
use super::super::super::types::SoundSpeedShiftImageView;
use super::super::types::{
    SoundSpeedShiftBatchStreamSummary, SoundSpeedShiftFrameSummary, SoundSpeedShiftPlan,
    SoundSpeedShiftPlanWorkspace,
};
use super::super::validation::validate_frame_batch;
use super::solve::solve_batch_frame;
use super::summary::{objective_summary, stream_summary};

impl SoundSpeedShiftPlan {
    /// Reconstruct a batch and stream borrowed image views to a callback.
    ///
    /// This path allocates a temporary plan workspace and one temporary output
    /// image. It does not store batch frame records.
    ///
    /// # Errors
    /// Returns [`kwavers_core::error::KwaversError`] when the batch is empty,
    /// any frame violates the fixed acquisition row contract, or the callback
    /// returns an error.
    pub fn reconstruct_frames_streaming<F>(
        &self,
        frame_time_shifts_s: &[&[f64]],
        on_frame: F,
    ) -> KwaversResult<SoundSpeedShiftBatchStreamSummary>
    where
        F: for<'frame> FnMut(
            SoundSpeedShiftFrameSummary,
            SoundSpeedShiftImageView<'frame>,
        ) -> KwaversResult<()>,
    {
        let mut workspace = SoundSpeedShiftPlanWorkspace::new();
        self.reconstruct_frames_streaming_with_plan_workspace(
            frame_time_shifts_s,
            &mut workspace,
            on_frame,
        )
    }

    /// Reconstruct a batch through caller-owned plan buffers and stream views.
    ///
    /// The callback receives a summary and a borrowed image view for each
    /// solved frame. The view is valid only for the duration of the callback;
    /// the next frame reuses the same image and solver buffers.
    ///
    /// # Errors
    /// Returns [`kwavers_core::error::KwaversError`] when the batch is empty,
    /// any frame violates the fixed acquisition row contract, or the callback
    /// returns an error.
    pub fn reconstruct_frames_streaming_with_plan_workspace<F>(
        &self,
        frame_time_shifts_s: &[&[f64]],
        workspace: &mut SoundSpeedShiftPlanWorkspace,
        mut on_frame: F,
    ) -> KwaversResult<SoundSpeedShiftBatchStreamSummary>
    where
        F: for<'frame> FnMut(
            SoundSpeedShiftFrameSummary,
            SoundSpeedShiftImageView<'frame>,
        ) -> KwaversResult<()>,
    {
        validate_frame_batch(frame_time_shifts_s, self.samples.len())?;
        workspace.sampled_rhs.resize(self.operator.rows(), 0.0);
        let mut output_image = Array2::<f64>::zeros(self.shape);

        for (frame_index, time_shifts_s) in frame_time_shifts_s.iter().enumerate() {
            solve_batch_frame(
                self,
                time_shifts_s,
                &mut workspace.sampled_rhs,
                &mut workspace.solver,
            );
            let summary = objective_summary(frame_index, &workspace.solver.objective_history)?;
            solved_image_from_operator_into(&self.operator, &workspace.solver, &mut output_image);
            let view = SoundSpeedShiftImageView {
                sound_speed_shift_m_s: &output_image,
                objective_history: &workspace.solver.objective_history,
                rows_used: self.operator.rows(),
                rows_available: self.samples.len(),
                active_voxels: self.operator.cols(),
                model_family: self.config.model_family(),
                sampling: self.config.sampling,
                prior: self.config.prior,
            };
            on_frame(summary, view)?;
        }

        Ok(stream_summary(self, frame_time_shifts_s.len()))
    }
}
