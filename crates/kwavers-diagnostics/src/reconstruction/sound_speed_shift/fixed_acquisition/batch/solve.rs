//! Shared fixed-acquisition batch solve primitive.

use super::super::super::solve_operator_frame_with_metrics;
use super::super::super::types::SoundSpeedShiftWorkspace;
use super::super::types::SoundSpeedShiftPlan;

pub(super) fn solve_batch_frame(
    plan: &SoundSpeedShiftPlan,
    time_shifts_s: &[f64],
    sampled_rhs: &mut [f64],
    workspace: &mut SoundSpeedShiftWorkspace,
) {
    plan.operator.rhs_from_time_shift_values_into(
        time_shifts_s,
        plan.config.reference_sound_speed_m_s,
        sampled_rhs,
    );
    solve_operator_frame_with_metrics(
        &plan.operator,
        sampled_rhs,
        plan.config,
        workspace,
        &plan.metrics,
    );
}
