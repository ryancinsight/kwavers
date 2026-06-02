//! Objective-only streaming for fixed-acquisition batch reconstruction.

use kwavers_core::error::KwaversResult;

use super::super::types::{
    SoundSpeedShiftBatchStreamSummary, SoundSpeedShiftFrameSummary, SoundSpeedShiftPlan,
    SoundSpeedShiftPlanWorkspace,
};
use super::super::validation::validate_frame_batch;
use super::solve::solve_batch_frame;
use super::summary::{objective_summary, stream_summary};

impl SoundSpeedShiftPlan {
    /// Reconstruct a batch and stream objective evidence to a callback.
    ///
    /// This path allocates a temporary plan workspace but does not materialize
    /// reconstructed images. It is intended for online convergence metrics,
    /// quality control, or frame rejection logic that only needs objective
    /// values.
    ///
    /// # Errors
    /// Returns [`kwavers_core::error::KwaversError`] when the batch is empty,
    /// any frame violates the fixed acquisition row contract, or the callback
    /// returns an error.
    pub fn reconstruct_frames_streaming_objectives<F>(
        &self,
        frame_time_shifts_s: &[&[f64]],
        on_frame: F,
    ) -> KwaversResult<SoundSpeedShiftBatchStreamSummary>
    where
        F: for<'frame> FnMut(SoundSpeedShiftFrameSummary, &'frame [f64]) -> KwaversResult<()>,
    {
        let mut workspace = SoundSpeedShiftPlanWorkspace::new();
        self.reconstruct_frames_streaming_objectives_with_plan_workspace(
            frame_time_shifts_s,
            &mut workspace,
            on_frame,
        )
    }

    /// Reconstruct a batch through caller-owned plan buffers and stream
    /// objective evidence without image materialization.
    ///
    /// The objective-history slice is borrowed from the solver workspace and
    /// remains valid only for the duration of the callback.
    ///
    /// # Errors
    /// Returns [`kwavers_core::error::KwaversError`] when the batch is empty,
    /// any frame violates the fixed acquisition row contract, or the callback
    /// returns an error.
    pub fn reconstruct_frames_streaming_objectives_with_plan_workspace<F>(
        &self,
        frame_time_shifts_s: &[&[f64]],
        workspace: &mut SoundSpeedShiftPlanWorkspace,
        mut on_frame: F,
    ) -> KwaversResult<SoundSpeedShiftBatchStreamSummary>
    where
        F: for<'frame> FnMut(SoundSpeedShiftFrameSummary, &'frame [f64]) -> KwaversResult<()>,
    {
        validate_frame_batch(frame_time_shifts_s, self.samples.len())?;
        workspace.sampled_rhs.resize(self.operator.rows(), 0.0);

        for (frame_index, time_shifts_s) in frame_time_shifts_s.iter().enumerate() {
            solve_batch_frame(
                self,
                time_shifts_s,
                &mut workspace.sampled_rhs,
                &mut workspace.solver,
            );
            let summary = objective_summary(frame_index, &workspace.solver.objective_history)?;
            on_frame(summary, &workspace.solver.objective_history)?;
        }

        Ok(stream_summary(self, frame_time_shifts_s.len()))
    }
}
