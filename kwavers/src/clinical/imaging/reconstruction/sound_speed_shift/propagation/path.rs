//! Discrete propagation paths for row assembly.

use std::f64::consts::TAU;

use crate::solver::inverse::same_aperture::PlanarPoint;

use super::super::types::{ShiftPropagation, SoundSpeedShiftSample};

#[derive(Clone, Copy, Debug)]
pub(super) struct PathSegment {
    pub start: PlanarPoint,
    pub end: PlanarPoint,
    pub length_m: f64,
    pub midpoint_distance_m: f64,
}

pub(super) fn build_path(
    sample: &SoundSpeedShiftSample,
    propagation: ShiftPropagation,
) -> Vec<PathSegment> {
    match propagation {
        ShiftPropagation::StraightRay => straight_segment(sample.transmitter, sample.receiver),
        ShiftPropagation::CircularArc {
            sagitta_m,
            segments,
        } => circular_arc_segments(sample.transmitter, sample.receiver, sagitta_m, segments),
    }
}

fn straight_segment(start: PlanarPoint, end: PlanarPoint) -> Vec<PathSegment> {
    let length_m = distance(start, end);
    vec![PathSegment {
        start,
        end,
        length_m,
        midpoint_distance_m: 0.5 * length_m,
    }]
}

fn circular_arc_segments(
    start: PlanarPoint,
    end: PlanarPoint,
    sagitta_m: f64,
    segments: usize,
) -> Vec<PathSegment> {
    let chord_m = distance(start, end);
    if chord_m <= f64::EPSILON {
        return Vec::new();
    }
    let mid = PlanarPoint {
        x_m: 0.5 * (start.x_m + end.x_m),
        y_m: 0.5 * (start.y_m + end.y_m),
    };
    let normal = PlanarPoint {
        x_m: -(end.y_m - start.y_m) / chord_m,
        y_m: (end.x_m - start.x_m) / chord_m,
    };
    let center_offset_m = 0.5 * sagitta_m - chord_m * chord_m / (8.0 * sagitta_m);
    let center = PlanarPoint {
        x_m: mid.x_m + center_offset_m * normal.x_m,
        y_m: mid.y_m + center_offset_m * normal.y_m,
    };
    let midpoint = PlanarPoint {
        x_m: mid.x_m + sagitta_m * normal.x_m,
        y_m: mid.y_m + sagitta_m * normal.y_m,
    };
    let radius_m = distance(center, start);
    let start_angle = angle(center, start);
    let end_angle = angle(center, end);
    let mid_angle = angle(center, midpoint);
    let delta = signed_arc_delta(start_angle, end_angle, mid_angle);
    let mut points = Vec::with_capacity(segments + 1);
    for idx in 0..=segments {
        let theta = start_angle + delta * idx as f64 / segments as f64;
        points.push(PlanarPoint {
            x_m: center.x_m + radius_m * theta.cos(),
            y_m: center.y_m + radius_m * theta.sin(),
        });
    }
    segments_from_points(&points)
}

fn segments_from_points(points: &[PlanarPoint]) -> Vec<PathSegment> {
    let mut out = Vec::with_capacity(points.len().saturating_sub(1));
    let mut cumulative_m = 0.0;
    for pair in points.windows(2) {
        let length_m = distance(pair[0], pair[1]);
        out.push(PathSegment {
            start: pair[0],
            end: pair[1],
            length_m,
            midpoint_distance_m: cumulative_m + 0.5 * length_m,
        });
        cumulative_m += length_m;
    }
    out
}

fn signed_arc_delta(start: f64, end: f64, midpoint: f64) -> f64 {
    let ccw = positive_delta(start, end);
    if angle_lies_on_ccw_arc(start, ccw, midpoint) {
        ccw
    } else {
        ccw - TAU
    }
}

fn angle_lies_on_ccw_arc(start: f64, delta: f64, angle: f64) -> bool {
    positive_delta(start, angle) <= delta + 1.0e-12
}

fn positive_delta(start: f64, end: f64) -> f64 {
    (end - start).rem_euclid(TAU)
}

fn angle(center: PlanarPoint, point: PlanarPoint) -> f64 {
    (point.y_m - center.y_m).atan2(point.x_m - center.x_m)
}

pub(super) fn distance(a: PlanarPoint, b: PlanarPoint) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}
