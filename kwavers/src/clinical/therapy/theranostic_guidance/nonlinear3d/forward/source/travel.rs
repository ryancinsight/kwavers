//! Ray-tracing travel time and source-cell iteration.

use crate::clinical::therapy::theranostic_guidance::nonlinear3d::types::{flat_index, GridIndex};
use crate::core::constants::fundamental::SOUND_SPEED_AIR;

use super::plan::SourcePlan;

/// Straight-ray slowness integral from `source` to `focus` on the speed map.
pub(super) fn source_focus_travel_time_s(
    speed: &[f64],
    n: usize,
    spacing_m: f64,
    source: GridIndex,
    focus: GridIndex,
) -> f64 {
    let dx = focus.x as f64 - source.x as f64;
    let dy = focus.y as f64 - source.y as f64;
    let dz = focus.z as f64 - source.z as f64;
    let steps = (dx.abs().max(dy.abs()).max(dz.abs()).ceil() as usize).max(1);
    let segment_length_m = spacing_m * dx.hypot(dy).hypot(dz) / steps as f64;
    let mut travel_time_s = 0.0;
    for step in 0..steps {
        let t = (step as f64 + 0.5) / steps as f64;
        let idx = GridIndex {
            x: (source.x as f64 + t * dx)
                .round()
                .clamp(0.0, (n - 1) as f64) as usize,
            y: (source.y as f64 + t * dy)
                .round()
                .clamp(0.0, (n - 1) as f64) as usize,
            z: (source.z as f64 + t * dz)
                .round()
                .clamp(0.0, (n - 1) as f64) as usize,
        };
        let c = speed[flat_index(idx, n)].max(SOUND_SPEED_AIR);
        travel_time_s += segment_length_m / c;
    }
    travel_time_s
}

/// Iterator over every grid cell referenced by any source stencil.
pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d::forward) fn source_cells(
    source_plan: &SourcePlan,
) -> impl Iterator<Item = usize> + '_ {
    source_plan
        .source_stencils
        .iter()
        .flat_map(|stencil| stencil.iter().map(|(cell, _)| *cell))
}
