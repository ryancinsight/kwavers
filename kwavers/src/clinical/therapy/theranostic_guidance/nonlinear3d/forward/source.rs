//! Source geometry, encoding, signal generation, and grid injection.

use super::super::encoding::{focused_delay_s, max_source_focus_distance_m, SourceEncoding};
use super::super::types::{flat_index, Nonlinear3dAperture, Nonlinear3dConfig};
use super::TimeSchedule;

pub(super) struct SourcePlan {
    pub(super) source_cells: Vec<usize>,
    pub(super) max_focus_distance: f64,
    pub(super) reference_speed: f64,
    pub(super) weight_norm: f64,
}

pub(super) struct DriveContext<'a> {
    pub(super) aperture: &'a Nonlinear3dAperture,
    pub(super) config: &'a Nonlinear3dConfig,
    pub(super) schedule: TimeSchedule,
    pub(super) encoding: SourceEncoding,
    pub(super) spacing_m: f64,
    pub(super) max_focus_distance: f64,
    pub(super) reference_speed: f64,
    pub(super) weight_norm: f64,
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
    let weight_norm = aperture
        .sources
        .iter()
        .enumerate()
        .map(|(source, _)| encoding.source_weight(source, aperture.sources.len()).abs())
        .sum::<f64>()
        .max(1.0);
    SourcePlan {
        source_cells,
        max_focus_distance,
        reference_speed,
        weight_norm,
    }
}

pub(super) fn inject_sources(
    next: &mut [f64],
    source_cells: &[usize],
    drive: &DriveContext<'_>,
    step: usize,
) {
    let time = step as f64 * drive.schedule.dt_s;
    for (source_idx, cell) in source_cells.iter().enumerate() {
        let delay = focused_delay_s(
            drive.aperture.sources[source_idx],
            drive.aperture.focus,
            drive.max_focus_distance,
            drive.spacing_m,
            drive.reference_speed,
        );
        let signal = source_signal(time - delay, drive.config)
            * drive
                .encoding
                .source_weight(source_idx, drive.aperture.sources.len())
            / drive.weight_norm;
        next[*cell] += signal;
    }
}

fn source_signal(t: f64, config: &Nonlinear3dConfig) -> f64 {
    let duration = config.cycles / config.frequency_hz;
    if t < 0.0 || t >= duration {
        return 0.0;
    }
    let phase = 2.0 * std::f64::consts::PI * config.frequency_hz * t;
    let window = (std::f64::consts::PI * t / duration).sin().powi(2);
    config.source_pressure_pa * phase.sin() * window
}
