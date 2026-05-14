//! Finite-frequency Fresnel-tube voxel sensitivity.

use crate::solver::inverse::same_aperture::PlanarPoint;

use super::super::ray::{cell_center, grid_bounds};
use super::{active_column, path::distance, path::PathSegment};

pub(super) fn entries(
    path: &[PathSegment],
    active_lookup: &[Option<usize>],
    shape: (usize, usize),
    spacing_m: f64,
    wavelength_m: f64,
    support_radius_m: f64,
) -> Vec<(usize, f64)> {
    let total_length_m = path.iter().map(|segment| segment.length_m).sum::<f64>();
    if total_length_m <= f64::EPSILON {
        return Vec::new();
    }

    let mut out = Vec::new();
    for segment in path {
        let sigma_m = fresnel_sigma(
            wavelength_m,
            total_length_m,
            segment.midpoint_distance_m,
            spacing_m,
        );
        let radius_m = support_radius_m.min(3.0 * sigma_m).max(0.5 * spacing_m);
        let mut local =
            finite_segment_candidates(segment, active_lookup, shape, spacing_m, sigma_m, radius_m);
        let raw_sum = local.iter().map(|(_, value)| *value).sum::<f64>();
        if raw_sum <= f64::EPSILON {
            continue;
        }
        for (_, value) in &mut local {
            *value *= segment.length_m / raw_sum;
        }
        out.extend(local);
    }
    out
}

fn finite_segment_candidates(
    segment: &PathSegment,
    active_lookup: &[Option<usize>],
    shape: (usize, usize),
    spacing_m: f64,
    sigma_m: f64,
    radius_m: f64,
) -> Vec<(usize, f64)> {
    let (nx, ny) = shape;
    let (xmin, _, ymin, _) = grid_bounds(shape, spacing_m);
    let ix_range = cell_range(
        segment.start.x_m.min(segment.end.x_m) - radius_m,
        segment.start.x_m.max(segment.end.x_m) + radius_m,
        xmin,
        nx,
        spacing_m,
    );
    let iy_range = cell_range(
        segment.start.y_m.min(segment.end.y_m) - radius_m,
        segment.start.y_m.max(segment.end.y_m) + radius_m,
        ymin,
        ny,
        spacing_m,
    );
    let mut out = Vec::new();
    for ix in ix_range.0..=ix_range.1 {
        for iy in iy_range.0..=iy_range.1 {
            let center = cell_center(ix, iy, shape, spacing_m);
            let transverse_m = point_segment_distance(center, segment.start, segment.end);
            if transverse_m <= radius_m {
                let weight = (-0.5 * (transverse_m / sigma_m).powi(2)).exp();
                if let Some(entry) = active_column(active_lookup, ix, iy, ny, weight) {
                    out.push(entry);
                }
            }
        }
    }
    out
}

fn fresnel_sigma(
    wavelength_m: f64,
    total_length_m: f64,
    midpoint_distance_m: f64,
    spacing_m: f64,
) -> f64 {
    let axial_m = midpoint_distance_m * (total_length_m - midpoint_distance_m) / total_length_m;
    (wavelength_m * axial_m).sqrt().max(0.5 * spacing_m)
}

fn cell_range(
    lower_m: f64,
    upper_m: f64,
    grid_lower_m: f64,
    cells: usize,
    spacing_m: f64,
) -> (usize, usize) {
    let first = (((lower_m - grid_lower_m) / spacing_m).floor().max(0.0) as usize).min(cells - 1);
    let last = (((upper_m - grid_lower_m) / spacing_m).floor().max(0.0) as usize).min(cells - 1);
    (first.min(last), first.max(last))
}

fn point_segment_distance(point: PlanarPoint, start: PlanarPoint, end: PlanarPoint) -> f64 {
    let vx = end.x_m - start.x_m;
    let vy = end.y_m - start.y_m;
    let length_sq = vx * vx + vy * vy;
    if length_sq <= f64::EPSILON {
        return distance(point, start);
    }
    let wx = point.x_m - start.x_m;
    let wy = point.y_m - start.y_m;
    let t = ((wx * vx + wy * vy) / length_sq).clamp(0.0, 1.0);
    let projection = PlanarPoint {
        x_m: start.x_m + t * vx,
        y_m: start.y_m + t * vy,
    };
    distance(point, projection)
}
