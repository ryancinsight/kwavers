use std::f64::consts::PI;

use ndarray::Array2;

use crate::core::error::KwaversResult;
use crate::domain::source::transducers::focused::BowlTransducer;

use super::super::aperture::{
    abdominal_aperture_frame, abdominal_arc_point_2d, abdominal_arc_spec,
    abdominal_imaging_point_2d,
};
use super::super::config::{AnatomyKind, TheranosticInverseConfig};
use super::super::skin::nearest_external_skin_point;
use super::boundary::is_boundary_2d;
use super::types::Point2;

#[derive(Clone, Debug)]
pub struct DeviceLayout {
    pub therapy_elements: Vec<Point2>,
    pub imaging_receivers: Vec<Point2>,
    pub focus_m: Point2,
    pub skin_contact_m: Point2,
    pub model_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DevicePlacementMetrics {
    pub min_body_clearance_m: f64,
    pub mean_body_clearance_m: f64,
    pub max_body_clearance_m: f64,
    pub skin_contact_to_nearest_aperture_m: f64,
}

pub fn build_device_layout(
    config: &TheranosticInverseConfig,
    body_mask: &Array2<bool>,
    target_mask: &Array2<bool>,
    spacing_m: f64,
) -> KwaversResult<DeviceLayout> {
    let focus = centroid_or_center(target_mask, body_mask, spacing_m);
    let layout = match config.anatomy {
        AnatomyKind::Brain => {
            transcranial_focused_bowl_projection_layout(config, body_mask, spacing_m, focus)?
        }
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            abdominal_layout(config, body_mask, spacing_m, focus)?
        }
    };
    Ok(layout)
}

fn transcranial_focused_bowl_projection_layout(
    config: &TheranosticInverseConfig,
    body_mask: &Array2<bool>,
    spacing_m: f64,
    focus: Point2,
) -> KwaversResult<DeviceLayout> {
    let radius = body_radius_from_focus(body_mask, spacing_m, focus) + 0.015;
    let actual_radius = radius.max(config.focal_radius_m);
    let elements = BowlTransducer::transverse_projection_ring(
        [focus.x_m, focus.y_m],
        actual_radius,
        config.element_count,
    )?
    .into_iter()
    .map(|point| Point2 {
        x_m: point[0],
        y_m: point[1],
    })
    .collect();
    Ok(DeviceLayout {
        therapy_elements: elements,
        imaging_receivers: Vec::new(),
        focus_m: focus,
        skin_contact_m: nearest_body_boundary_point(body_mask, spacing_m, focus),
        model_name: "transcranial_focused_bowl_projection".to_owned(),
    })
}

fn body_radius_from_focus(mask: &Array2<bool>, spacing_m: f64, focus: Point2) -> f64 {
    let center = centered_origin(mask);
    mask.indexed_iter()
        .filter(|(_, active)| **active)
        .map(|((ix, iy), _)| {
            let x = (ix as f64 - center.0) * spacing_m;
            let y = (iy as f64 - center.1) * spacing_m;
            (x - focus.x_m).hypot(y - focus.y_m)
        })
        .fold(0.0, f64::max)
}

fn abdominal_layout(
    config: &TheranosticInverseConfig,
    body_mask: &Array2<bool>,
    spacing_m: f64,
    focus: Point2,
) -> KwaversResult<DeviceLayout> {
    let skin = nearest_external_skin_point(body_mask, spacing_m, spacing_m, focus.x_m, focus.y_m)?;
    let skin = Point2 {
        x_m: skin.x_m,
        y_m: skin.y_m,
    };
    let frame = abdominal_aperture_frame(focus.x_m, focus.y_m, skin.x_m, skin.y_m);
    let arc = abdominal_arc_spec(config, frame.depth_m);
    let left = config.element_count / 2;
    let right = config.element_count - left;
    let mut elements = Vec::with_capacity(config.element_count);
    for idx in 0..left {
        let t = if left > 1 {
            idx as f64 / (left - 1) as f64
        } else {
            0.0
        };
        elements.push(abdominal_arc_point(
            frame,
            focus,
            arc.radius_m,
            -arc.half_angle_rad,
            -arc.cutout_angle_rad,
            t,
        ));
    }
    for idx in 0..right {
        let t = if right > 1 {
            idx as f64 / (right - 1) as f64
        } else {
            0.0
        };
        elements.push(abdominal_arc_point(
            frame,
            focus,
            arc.radius_m,
            arc.cutout_angle_rad,
            arc.half_angle_rad,
            t,
        ));
    }
    let receiver_count = 64;
    let imaging_receivers = (0..receiver_count)
        .map(|idx| {
            let t = if receiver_count > 1 {
                idx as f64 / (receiver_count - 1) as f64
            } else {
                0.5
            };
            let (x_m, y_m) =
                abdominal_imaging_point_2d(frame, skin.x_m, skin.y_m, config.central_cutout_m, t);
            Point2 { x_m, y_m }
        })
        .collect();
    Ok(DeviceLayout {
        therapy_elements: elements,
        imaging_receivers,
        focus_m: focus,
        skin_contact_m: skin,
        model_name: "histosonics_like_256_element_skin_coupled_arc".to_owned(),
    })
}

fn abdominal_arc_point(
    frame: super::super::aperture::AbdominalApertureFrame,
    focus: Point2,
    radius: f64,
    a: f64,
    b: f64,
    t: f64,
) -> Point2 {
    let theta = a + (b - a) * t;
    let (x_m, y_m) = abdominal_arc_point_2d(frame, focus.x_m, focus.y_m, radius, theta);
    Point2 { x_m, y_m }
}

fn centroid_or_center(mask: &Array2<bool>, fallback: &Array2<bool>, spacing_m: f64) -> Point2 {
    let selected = if mask.iter().any(|v| *v) {
        mask
    } else {
        fallback
    };
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0.0;
    let center = centered_origin(selected);
    for ((ix, iy), active) in selected.indexed_iter() {
        if *active {
            sx += (ix as f64 - center.0) * spacing_m;
            sy += (iy as f64 - center.1) * spacing_m;
            n += 1.0;
        }
    }
    if n > 0.0 {
        Point2 {
            x_m: sx / n,
            y_m: sy / n,
        }
    } else {
        Point2 { x_m: 0.0, y_m: 0.0 }
    }
}

fn nearest_body_point(mask: &Array2<bool>, spacing_m: f64, focus: Point2) -> Point2 {
    let center = centered_origin(mask);
    let mut best = Point2 { x_m: 0.0, y_m: 0.0 };
    let mut best_distance = f64::INFINITY;
    for ((ix, iy), active) in mask.indexed_iter() {
        if *active {
            let point = Point2 {
                x_m: (ix as f64 - center.0) * spacing_m,
                y_m: (iy as f64 - center.1) * spacing_m,
            };
            let d = (point.x_m - focus.x_m).hypot(point.y_m - focus.y_m);
            if d < best_distance {
                best = point;
                best_distance = d;
            }
        }
    }
    best
}

fn nearest_body_boundary_point(mask: &Array2<bool>, spacing_m: f64, focus: Point2) -> Point2 {
    let center = centered_origin(mask);
    let mut best = Point2 { x_m: 0.0, y_m: 0.0 };
    let mut best_distance = f64::INFINITY;
    for ((ix, iy), active) in mask.indexed_iter() {
        if *active && is_boundary_2d(mask, ix, iy) {
            let point = Point2 {
                x_m: (ix as f64 - center.0) * spacing_m,
                y_m: (iy as f64 - center.1) * spacing_m,
            };
            let d = (point.x_m - focus.x_m).hypot(point.y_m - focus.y_m);
            if d < best_distance {
                best = point;
                best_distance = d;
            }
        }
    }
    if best_distance.is_finite() {
        best
    } else {
        nearest_body_point(mask, spacing_m, focus)
    }
}

fn centered_origin(mask: &Array2<bool>) -> (f64, f64) {
    let (nx, ny) = mask.dim();
    ((nx - 1) as f64 * 0.5, (ny - 1) as f64 * 0.5)
}

#[must_use]
pub fn placement_metrics(
    layout: &DeviceLayout,
    body_mask: &Array2<bool>,
    spacing_m: f64,
) -> DevicePlacementMetrics {
    let center = centered_origin(body_mask);
    let body_points = body_mask
        .indexed_iter()
        .filter_map(|((ix, iy), active)| {
            active.then_some(Point2 {
                x_m: (ix as f64 - center.0) * spacing_m,
                y_m: (iy as f64 - center.1) * spacing_m,
            })
        })
        .collect::<Vec<_>>();
    let aperture_points = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .copied()
        .collect::<Vec<_>>();
    let mut min_clearance = f64::INFINITY;
    let mut max_clearance: f64 = 0.0;
    let mut sum_clearance: f64 = 0.0;
    let mut count: f64 = 0.0;
    for aperture in &aperture_points {
        let clearance = body_points
            .iter()
            .map(|body| distance(*aperture, *body))
            .fold(f64::INFINITY, f64::min);
        if clearance.is_finite() {
            min_clearance = min_clearance.min(clearance);
            max_clearance = max_clearance.max(clearance);
            sum_clearance += clearance;
            count += 1.0;
        }
    }
    let skin_contact_to_nearest_aperture_m = aperture_points
        .iter()
        .map(|aperture| distance(*aperture, layout.skin_contact_m))
        .fold(f64::INFINITY, f64::min);
    DevicePlacementMetrics {
        min_body_clearance_m: if min_clearance.is_finite() {
            min_clearance
        } else {
            0.0
        },
        mean_body_clearance_m: if count > 0.0 {
            sum_clearance / count
        } else {
            0.0
        },
        max_body_clearance_m: max_clearance,
        skin_contact_to_nearest_aperture_m: if skin_contact_to_nearest_aperture_m.is_finite() {
            skin_contact_to_nearest_aperture_m
        } else {
            0.0
        },
    }
}

pub fn distance(a: Point2, b: Point2) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}

#[must_use]
pub fn angle_span(layout: &DeviceLayout) -> f64 {
    if layout.therapy_elements.len() < 2 {
        return 0.0;
    }
    let mut min_angle = PI;
    let mut max_angle = -PI;
    for element in &layout.therapy_elements {
        let angle = (element.y_m - layout.focus_m.y_m).atan2(element.x_m - layout.focus_m.x_m);
        min_angle = min_angle.min(angle);
        max_angle = max_angle.max(angle);
    }
    max_angle - min_angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn transcranial_projection_ring_is_focus_relative() {
        let spacing_m = 0.001;
        let focus = Point2 {
            x_m: 0.006,
            y_m: -0.004,
        };
        let mut body_mask = Array2::<bool>::from_elem((31, 29), false);
        for x in 0..31 {
            for y in 0..29 {
                let active =
                    ((x as f64 - 15.0) / 12.0).powi(2) + ((y as f64 - 14.0) / 10.0).powi(2) <= 1.0;
                body_mask[[x, y]] = active;
            }
        }
        let mut config = TheranosticInverseConfig::new(AnatomyKind::Brain);
        config.element_count = 32;
        config.focal_radius_m = 0.0;

        let layout =
            transcranial_focused_bowl_projection_layout(&config, &body_mask, spacing_m, focus)
                .unwrap();
        let expected_radius = body_radius_from_focus(&body_mask, spacing_m, focus) + 0.015;

        assert_eq!(layout.therapy_elements.len(), config.element_count);
        for element in &layout.therapy_elements {
            assert_close(distance(*element, focus), expected_radius);
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 64.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}
