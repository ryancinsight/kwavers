//! Shared fixed-acquisition batch solve primitive.

use super::super::super::solve_operator_frame_with_metrics;
use super::super::super::types::SoundSpeedShiftWorkspace;
use super::super::types::SoundSpeedShiftPlan;
use aequitas::systems::si::quantities::Time;

pub(super) fn solve_batch_frame(
    plan: &SoundSpeedShiftPlan,
    time_shifts: &[Time],
    sampled_rhs: &mut [f64],
    workspace: &mut SoundSpeedShiftWorkspace,
) {
    plan.operator.rhs_from_time_shift_values_into(
        time_shifts,
        plan.config.reference_sound_speed,
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
