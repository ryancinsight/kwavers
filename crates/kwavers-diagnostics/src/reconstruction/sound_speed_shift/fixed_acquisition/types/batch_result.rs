//! Fixed-acquisition batch result types.

use ndarray::Array2;

use super::super::super::types::{ShiftPrior, ShiftSampling};

/// Compact objective summary for one batch frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundSpeedShiftFrameSummary {
    /// Zero-based frame index in the submitted batch.
    pub frame_index: usize,
    /// Objective at the initial iterate.
    pub objective_initial: f64,
    /// Objective at the final retained iterate.
    pub objective_final: f64,
    /// Number of solver updates completed after the initial state.
    pub objective_iterations: usize,
}

/// Reconstructed image and objective evidence for one batch frame.
#[derive(Clone, Debug)]
pub struct SoundSpeedShiftBatchFrame {
    /// Estimated `delta c = c - c0` [m/s] for this frame when retained.
    pub sound_speed_shift_m_s: Option<Array2<f64>>,
    /// Compact objective summary.
    pub summary: SoundSpeedShiftFrameSummary,
    /// Full objective history when requested by
    /// [`super::batch_config::SoundSpeedShiftObjectiveHistoryPolicy::Full`].
    pub objective_history: Vec<f64>,
}

/// Aggregate result for a streamed fixed-acquisition batch reconstruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundSpeedShiftBatchStreamSummary {
    /// Number of frames reconstructed and passed to the callback.
    pub frames_processed: usize,
    /// Number of selected measurement rows used by each frame solve.
    pub rows_used: usize,
    /// Number of supplied measurement rows before sampling.
    pub rows_available: usize,
    /// Number of active image pixels in the reconstruction support.
    pub active_voxels: usize,
    /// Model identifier for audit trails.
    pub model_family: &'static str,
    /// Measurement-row policy used by the plan.
    pub sampling: ShiftSampling,
    /// Image prior used by the plan.
    pub prior: ShiftPrior,
}

/// Batch reconstruction result for one fixed acquisition plan.
#[derive(Clone, Debug)]
pub struct SoundSpeedShiftBatch {
    /// Per-frame reconstructed images and objective summaries.
    pub frames: Vec<SoundSpeedShiftBatchFrame>,
    /// Number of selected measurement rows used by each frame solve.
    pub rows_used: usize,
    /// Number of supplied measurement rows before sampling.
    pub rows_available: usize,
    /// Number of active image pixels in the reconstruction support.
    pub active_voxels: usize,
    /// Model identifier for audit trails.
    pub model_family: &'static str,
    /// Measurement-row policy used by the plan.
    pub sampling: ShiftSampling,
    /// Image prior used by the plan.
    pub prior: ShiftPrior,
}
