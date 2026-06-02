//! Limited-view top/bottom probe acquisition rows.

use kwavers_solver::inverse::same_aperture::PlanarPoint;

use super::super::SoundSpeedShiftSample;
use super::types::OpenProsShiftBenchmarkConfig;

pub(super) fn samples(config: &OpenProsShiftBenchmarkConfig) -> Vec<SoundSpeedShiftSample> {
    let shape = config.shape();
    let spacing_m = config.spacing_m();
    let axial_extent_m = shape.0 as f64 * spacing_m;
    let lateral_extent_m = shape.1 as f64 * spacing_m;
    let body_x_m = -0.5 * axial_extent_m;
    let rectal_x_m = 0.5 * axial_extent_m;
    let body_sources = source_line(body_x_m, lateral_extent_m, config.source_count_per_probe);
    let rectal_sources = source_line(rectal_x_m, lateral_extent_m, config.source_count_per_probe);
    let body_receivers = receiver_line(body_x_m, lateral_extent_m, config.receiver_count_per_probe);
    let rectal_receivers = receiver_line(
        rectal_x_m,
        lateral_extent_m,
        config.receiver_count_per_probe,
    );

    let mut out =
        Vec::with_capacity(4 * config.source_count_per_probe * config.receiver_count_per_probe);
    append_probe_pair(&mut out, &body_sources, &body_receivers);
    append_probe_pair(&mut out, &body_sources, &rectal_receivers);
    append_probe_pair(&mut out, &rectal_sources, &rectal_receivers);
    append_probe_pair(&mut out, &rectal_sources, &body_receivers);
    out
}

fn append_probe_pair(
    out: &mut Vec<SoundSpeedShiftSample>,
    transmitters: &[PlanarPoint],
    receivers: &[PlanarPoint],
) {
    for transmitter in transmitters {
        for receiver in receivers {
            out.push(SoundSpeedShiftSample::new(*transmitter, *receiver, 0.0));
        }
    }
}

fn source_line(x_m: f64, lateral_extent_m: f64, count: usize) -> Vec<PlanarPoint> {
    if count == 0 {
        return Vec::new();
    }
    let first = -0.5 * lateral_extent_m;
    let pitch = lateral_extent_m / count as f64;
    const NONALIASING_SOURCE_OFFSET: f64 = 0.37;
    (0..count)
        .map(|idx| PlanarPoint {
            x_m,
            y_m: first + (idx as f64 + NONALIASING_SOURCE_OFFSET) * pitch,
        })
        .collect()
}

fn receiver_line(x_m: f64, lateral_extent_m: f64, count: usize) -> Vec<PlanarPoint> {
    match count {
        0 => Vec::new(),
        1 => vec![PlanarPoint { x_m, y_m: 0.0 }],
        _ => {
            let first = -0.5 * lateral_extent_m;
            let pitch = lateral_extent_m / (count - 1) as f64;
            (0..count)
                .map(|idx| PlanarPoint {
                    x_m,
                    y_m: first + idx as f64 * pitch,
                })
                .collect()
        }
    }
}
