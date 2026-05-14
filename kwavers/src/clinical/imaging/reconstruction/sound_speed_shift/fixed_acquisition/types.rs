//! Fixed-acquisition plan state.

use ndarray::Array2;

use super::super::operator::SoundSpeedShiftOperator;
use super::super::types::{
    ShiftPrior, ShiftSampling, SoundSpeedShiftConfig, SoundSpeedShiftSample,
};

/// Cached geometry and operator for repeated speed-shift frames.
///
/// A plan fixes the acquisition geometry, active mask, sampling policy,
/// propagation model, and sensitivity model. Per-frame measured shifts are
/// supplied as a plain slice in the original acquisition row order.
#[derive(Clone, Debug)]
pub struct SoundSpeedShiftPlan {
    pub(super) samples: Vec<SoundSpeedShiftSample>,
    pub(super) operator: SoundSpeedShiftOperator,
    pub(super) config: SoundSpeedShiftConfig,
    pub(super) shape: (usize, usize),
}

/// Objective-history retention policy for batch reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoundSpeedShiftObjectiveHistoryPolicy {
    /// Store only initial/final objective values and iteration count.
    Compact,
    /// Store the full objective history for every frame.
    Full,
}

/// Batch reconstruction output policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoundSpeedShiftBatchConfig {
    /// Per-frame objective-history retention policy.
    pub objective_history: SoundSpeedShiftObjectiveHistoryPolicy,
}

impl Default for SoundSpeedShiftBatchConfig {
    fn default() -> Self {
        Self {
            objective_history: SoundSpeedShiftObjectiveHistoryPolicy::Compact,
        }
    }
}

impl SoundSpeedShiftBatchConfig {
    /// Select per-frame objective-history retention.
    #[must_use]
    pub fn with_objective_history(
        mut self,
        objective_history: SoundSpeedShiftObjectiveHistoryPolicy,
    ) -> Self {
        self.objective_history = objective_history;
        self
    }
}

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
    /// Estimated `delta c = c - c0` [m/s] for this frame.
    pub sound_speed_shift_m_s: Array2<f64>,
    /// Compact objective summary.
    pub summary: SoundSpeedShiftFrameSummary,
    /// Full objective history when requested by
    /// [`SoundSpeedShiftObjectiveHistoryPolicy::Full`].
    pub objective_history: Vec<f64>,
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
