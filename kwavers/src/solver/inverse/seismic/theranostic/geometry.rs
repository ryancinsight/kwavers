//! Source and receiver placement for theranostic ultrasound arrays.

use std::f64::consts::{PI, TAU};

use ndarray::Array2;

use super::config::{AnatomyKind, TheranosticFwiConfig};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2 {
    pub x_m: f64,
    pub y_m: f64,
}

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
    config: &TheranosticFwiConfig,
    body_mask: &Array2<bool>,
    target_mask: &Array2<bool>,
    spacing_m: f64,
) -> DeviceLayout {
    let focus = centroid_or_center(target_mask, body_mask, spacing_m);
    match config.anatomy {
        AnatomyKind::Brain => helmet_layout(config, body_mask, spacing_m, focus),
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            abdominal_layout(config, body_mask, spacing_m, focus)
        }
    }
}

fn helmet_layout(
    config: &TheranosticFwiConfig,
    body_mask: &Array2<bool>,
    spacing_m: f64,
    focus: Point2,
) -> DeviceLayout {
    let radius = body_radius(body_mask, spacing_m) + 0.015;
    let actual_radius = radius.max(config.focal_radius_m);
    let elements = (0..config.element_count)
        .map(|idx| {
            let theta = TAU * idx as f64 / config.element_count as f64;
            Point2 {
                x_m: actual_radius * theta.cos(),
                y_m: actual_radius * theta.sin(),
            }
        })
        .collect();
    DeviceLayout {
        therapy_elements: elements,
        imaging_receivers: Vec::new(),
        focus_m: focus,
        skin_contact_m: nearest_body_boundary_point(body_mask, spacing_m, focus),
        model_name: "insightec_like_1024_element_helmet_projection".to_owned(),
    }
}

fn abdominal_layout(
    config: &TheranosticFwiConfig,
    body_mask: &Array2<bool>,
    spacing_m: f64,
    focus: Point2,
) -> DeviceLayout {
    let skin = anterior_skin_contact(body_mask, spacing_m, focus);
    let radius = config
        .focal_radius_m
        .max((focus.x_m - skin.x_m).abs() + 0.02);
    let half_angle = (config.lateral_extent_m / (2.0 * radius)).min(0.96).asin();
    let cutout = (config.central_cutout_m / (2.0 * radius)).min(0.90).asin();
    let left = config.element_count / 2;
    let right = config.element_count - left;
    let mut elements = Vec::with_capacity(config.element_count);
    for idx in 0..left {
        let t = if left > 1 {
            idx as f64 / (left - 1) as f64
        } else {
            0.0
        };
        elements.push(abdominal_arc_point(focus, radius, -half_angle, -cutout, t));
    }
    for idx in 0..right {
        let t = if right > 1 {
            idx as f64 / (right - 1) as f64
        } else {
            0.0
        };
        elements.push(abdominal_arc_point(focus, radius, cutout, half_angle, t));
    }
    let receiver_count = 64;
    let imaging_receivers = (0..receiver_count)
        .map(|idx| {
            let t = if receiver_count > 1 {
                idx as f64 / (receiver_count - 1) as f64
            } else {
                0.5
            };
            Point2 {
                x_m: skin.x_m - 0.003,
                y_m: skin.y_m + (t - 0.5) * config.central_cutout_m,
            }
        })
        .collect();
    DeviceLayout {
        therapy_elements: elements,
        imaging_receivers,
        focus_m: focus,
        skin_contact_m: skin,
        model_name: "histosonics_like_256_element_skin_coupled_arc".to_owned(),
    }
}

fn abdominal_arc_point(focus: Point2, radius: f64, a: f64, b: f64, t: f64) -> Point2 {
    let theta = a + (b - a) * t;
    Point2 {
        x_m: focus.x_m - radius * theta.cos(),
        y_m: focus.y_m + radius * theta.sin(),
    }
}

pub fn receiver_index(source: usize, offset: usize, count: usize) -> usize {
    (source + offset) % count
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

fn body_radius(mask: &Array2<bool>, spacing_m: f64) -> f64 {
    let center = centered_origin(mask);
    mask.indexed_iter()
        .filter_map(|((ix, iy), active)| {
            active.then(|| {
                let x = (ix as f64 - center.0) * spacing_m;
                let y = (iy as f64 - center.1) * spacing_m;
                (x * x + y * y).sqrt()
            })
        })
        .fold(0.0, f64::max)
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
            let distance = (point.x_m - focus.x_m).hypot(point.y_m - focus.y_m);
            if distance < best_distance {
                best = point;
                best_distance = distance;
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
        if *active && is_boundary(mask, ix, iy) {
            let point = Point2 {
                x_m: (ix as f64 - center.0) * spacing_m,
                y_m: (iy as f64 - center.1) * spacing_m,
            };
            let distance = (point.x_m - focus.x_m).hypot(point.y_m - focus.y_m);
            if distance < best_distance {
                best = point;
                best_distance = distance;
            }
        }
    }
    if best_distance.is_finite() {
        best
    } else {
        nearest_body_point(mask, spacing_m, focus)
    }
}

fn anterior_skin_contact(mask: &Array2<bool>, spacing_m: f64, focus: Point2) -> Point2 {
    let (nx, ny) = mask.dim();
    let center = centered_origin(mask);
    let iy = ((focus.y_m / spacing_m) + center.1).round() as isize;
    let iy = iy.clamp(0, ny.saturating_sub(1) as isize) as usize;
    for ix in 0..nx {
        if mask[[ix, iy]] {
            return Point2 {
                x_m: (ix as f64 - center.0) * spacing_m,
                y_m: (iy as f64 - center.1) * spacing_m,
            };
        }
    }
    Point2 {
        x_m: -0.5 * nx as f64 * spacing_m,
        y_m: focus.y_m,
    }
}

fn is_boundary(mask: &Array2<bool>, ix: usize, iy: usize) -> bool {
    let (nx, ny) = mask.dim();
    ix == 0
        || iy == 0
        || ix + 1 == nx
        || iy + 1 == ny
        || !mask[[ix - 1, iy]]
        || !mask[[ix + 1, iy]]
        || !mask[[ix, iy - 1]]
        || !mask[[ix, iy + 1]]
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

fn distance(a: Point2, b: Point2) -> f64 {
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
