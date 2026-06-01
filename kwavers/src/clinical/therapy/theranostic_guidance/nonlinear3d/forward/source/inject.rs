//! Grid pressure injection for all source elements at a single time step.

use crate::clinical::therapy::theranostic_guidance::nonlinear3d::types::Nonlinear3dConfig;

use super::plan::{DriveContext, SourcePlan};
use crate::core::constants::numerical::TWO_PI;

pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d::forward) fn inject_sources(
    next: &mut [f64],
    source_plan: &SourcePlan,
    drive: &DriveContext<'_>,
    step: usize,
) {
    let time = step as f64 * drive.schedule.dt_s;
    for ((stencil, delay), weight) in source_plan
        .source_stencils
        .iter()
        .zip(source_plan.focused_delays_s.iter())
        .zip(source_plan.encoding_weights.iter())
    {
        let signal = source_signal(time - delay, drive.config, drive.source_scale) * weight;
        let pressure_limit = drive.config.source_pressure_pa * drive.source_scale;
        for (cell, cell_weight) in stencil {
            next[*cell] =
                (next[*cell] + signal * cell_weight).clamp(-pressure_limit, pressure_limit);
        }
    }
}

fn source_signal(t: f64, config: &Nonlinear3dConfig, source_scale: f64) -> f64 {
    let duration = config.cycles / config.frequency_hz;
    if t < 0.0 || t >= duration {
        return 0.0;
    }
    let phase = TWO_PI * config.frequency_hz * t;
    let window = (std::f64::consts::PI * t / duration).sin().powi(2);
    config.source_pressure_pa * source_scale * phase.sin() * window
}
