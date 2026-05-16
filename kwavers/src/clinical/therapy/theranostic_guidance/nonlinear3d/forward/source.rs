//! Source geometry, encoding, signal generation, and grid injection.

use super::super::encoding::{focused_delay_s, max_source_focus_distance_m, SourceEncoding};
use super::super::types::{flat_index, Nonlinear3dAperture, Nonlinear3dConfig};
use super::TimeSchedule;

pub(super) struct SourcePlan {
    pub(super) source_cells: Vec<usize>,
    pub(super) focused_delays_s: Vec<f64>,
    pub(super) encoding_weights: Vec<f64>,
}

pub(super) struct DriveContext<'a> {
    pub(super) config: &'a Nonlinear3dConfig,
    pub(super) schedule: TimeSchedule,
    pub(super) source_scale: f64,
}

pub(super) fn build_source_plan(
    speed: &[f64],
    n: usize,
    spacing_m: f64,
    aperture: &Nonlinear3dAperture,
    encoding: SourceEncoding,
) -> SourcePlan {
    let source_cells = aperture
        .sources
        .iter()
        .map(|idx| flat_index(*idx, n))
        .collect::<Vec<_>>();
    let max_focus_distance =
        max_source_focus_distance_m(&aperture.sources, aperture.focus, spacing_m);
    let focus_cell = flat_index(aperture.focus, n);
    let reference_speed = speed[focus_cell].max(343.0);
    let raw_weights = aperture
        .sources
        .iter()
        .enumerate()
        .map(|(source, _)| encoding.source_weight(source, aperture.sources.len()))
        .collect::<Vec<_>>();
    let focused_delays_s = aperture
        .sources
        .iter()
        .map(|source| {
            focused_delay_s(
                *source,
                aperture.focus,
                max_focus_distance,
                spacing_m,
                reference_speed,
            )
        })
        .collect::<Vec<_>>();
    let encoding_weights = raw_weights
        .into_iter()
        .map(|weight| weight.signum())
        .collect::<Vec<_>>();
    SourcePlan {
        source_cells,
        focused_delays_s,
        encoding_weights,
    }
}

pub(super) fn inject_sources(
    next: &mut [f64],
    source_plan: &SourcePlan,
    drive: &DriveContext<'_>,
    step: usize,
) {
    let time = step as f64 * drive.schedule.dt_s;
    for ((cell, delay), weight) in source_plan
        .source_cells
        .iter()
        .zip(source_plan.focused_delays_s.iter())
        .zip(source_plan.encoding_weights.iter())
    {
        let signal = source_signal(time - delay, drive.config, drive.source_scale) * weight;
        next[*cell] += signal;
    }
}

fn source_signal(t: f64, config: &Nonlinear3dConfig, source_scale: f64) -> f64 {
    let duration = config.cycles / config.frequency_hz;
    if t < 0.0 || t >= duration {
        return 0.0;
    }
    let phase = 2.0 * std::f64::consts::PI * config.frequency_hz * t;
    let window = (std::f64::consts::PI * t / duration).sin().powi(2);
    config.source_pressure_pa * source_scale * phase.sin() * window
}

#[cfg(test)]
mod tests {
    use crate::clinical::therapy::theranostic_guidance::nonlinear3d::types::{
        GridIndex, Nonlinear3dAperture,
    };
    use crate::clinical::therapy::theranostic_guidance::Point3;

    use super::*;

    #[test]
    fn source_plan_preserves_per_element_drive_weights() {
        let n = 8;
        let cells = n * n * n;
        let aperture = Nonlinear3dAperture {
            sources: vec![
                GridIndex { x: 1, y: 1, z: 1 },
                GridIndex { x: 6, y: 1, z: 1 },
                GridIndex { x: 1, y: 6, z: 1 },
                GridIndex { x: 6, y: 6, z: 1 },
            ],
            receivers: Vec::new(),
            therapy_points_m: vec![
                Point3 {
                    x_m: 0.0,
                    y_m: 0.0,
                    z_m: 0.0,
                };
                4
            ],
            receiver_points_m: Vec::new(),
            model_name: "source_plan_test".to_owned(),
            focus: GridIndex { x: 4, y: 4, z: 4 },
        };
        let speed = vec![1500.0; cells];

        let plan = build_source_plan(
            &speed,
            n,
            1.0e-3,
            &aperture,
            SourceEncoding { index: 0, count: 1 },
        );

        assert_eq!(plan.encoding_weights, vec![1.0, 1.0, 1.0, 1.0]);
    }
}
