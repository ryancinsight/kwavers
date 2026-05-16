//! Fixed-acquisition batch result summaries.

use crate::core::error::{KwaversError, KwaversResult};

use super::super::types::{
    SoundSpeedShiftBatch, SoundSpeedShiftBatchFrame, SoundSpeedShiftBatchStreamSummary,
    SoundSpeedShiftFrameSummary, SoundSpeedShiftPlan,
};

pub(super) fn objective_summary(
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

pub(super) fn stream_summary(
    plan: &SoundSpeedShiftPlan,
    frames_processed: usize,
) -> SoundSpeedShiftBatchStreamSummary {
    SoundSpeedShiftBatchStreamSummary {
        frames_processed,
        rows_used: plan.operator.rows(),
        rows_available: plan.samples.len(),
        active_voxels: plan.operator.cols(),
        model_family: plan.config.model_family(),
        sampling: plan.config.sampling,
        prior: plan.config.prior,
    }
}

pub(super) fn retained_batch(
    plan: &SoundSpeedShiftPlan,
    frames: Vec<SoundSpeedShiftBatchFrame>,
) -> SoundSpeedShiftBatch {
    SoundSpeedShiftBatch {
        frames,
        rows_used: plan.operator.rows(),
        rows_available: plan.samples.len(),
        active_voxels: plan.operator.cols(),
        model_family: plan.config.model_family(),
        sampling: plan.config.sampling,
        prior: plan.config.prior,
    }
}
