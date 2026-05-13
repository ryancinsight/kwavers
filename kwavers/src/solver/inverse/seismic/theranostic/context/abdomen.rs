//! Full-slice abdominal placement context.

use std::collections::VecDeque;

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2, Array3};

use super::super::aperture::{abdominal_arc_spec, ABDOMINAL_SKIN_CLEARANCE_M};
use super::super::{medium::largest_target_slice, AnatomyKind, TheranosticFwiConfig};
use super::surface::{boundary_points_2d, surface_points_2d};
use super::{
    centered_origin_2d, centroid_2d, centroid_index, distance_2d, validate_spacing,
    PlacementContext, Point3,
};

pub fn build_abdominal_placement_context(
    anatomy: AnatomyKind,
    ct_volume_hu: &Array3<f64>,
    label_volume: &Array3<i16>,
    spacing_mm: [f64; 3],
    config: &TheranosticFwiConfig,
) -> KwaversResult<PlacementContext> {
    if ct_volume_hu.dim() != label_volume.dim() {
        return Err(KwaversError::InvalidInput(format!(
            "CT shape {:?} does not match segmentation shape {:?}",
            ct_volume_hu.dim(),
            label_volume.dim()
        )));
    }
    validate_spacing(spacing_mm)?;
    let slice_index = largest_target_slice(label_volume)?;
    let ct_slice = ct_volume_hu.slice(s![.., .., slice_index]).to_owned();
    let label_slice = label_volume.slice(s![.., .., slice_index]).to_owned();
    let target_mask = label_slice.mapv(|label| label == 2);
    let tissue = Array2::from_shape_fn(ct_slice.dim(), |idx| {
        ct_slice[idx] > -450.0 || label_slice[idx] > 0
    });
    let body_mask = connected_body_component(&tissue, &target_mask)?;
    let sx = spacing_mm[0] * 1.0e-3;
    let sy = spacing_mm[1] * 1.0e-3;
    let sz = spacing_mm[2] * 1.0e-3;
    let focus_2d = centroid_2d(&target_mask, sx, sy).ok_or_else(|| {
        KwaversError::InvalidInput("abdominal placement target mask is empty".to_owned())
    })?;
    let skin = left_skin_contact(&body_mask, sx, sy, focus_2d)?;
    let depth = (focus_2d.x_m - skin.x_m).max(0.0);
    let arc = abdominal_arc_spec(config, depth);
    let focus = Point3 {
        x_m: focus_2d.x_m,
        y_m: focus_2d.y_m,
        z_m: 0.0,
    };
    let therapy_points_m = abdominal_arc_points(
        config.element_count,
        focus,
        arc.radius_m,
        arc.half_angle_rad,
        arc.cutout_angle_rad,
    );
    let imaging_points_m = abdominal_imaging_points(skin, config.central_cutout_m, 64);
    let surface_stride = ((ct_slice.dim().0.max(ct_slice.dim().1) / 192).max(1)).min(6);
    let body_surface_points_m =
        surface_points_2d(&body_mask, sx, sy, sz, slice_index, surface_stride);

    Ok(PlacementContext {
        ct_hu: ct_slice,
        body_mask,
        target_mask,
        spacing_x_m: sx,
        spacing_y_m: sy,
        slice_index,
        therapy_points_m,
        imaging_points_m,
        body_surface_points_m,
        focus_m: focus,
        skin_contact_m: skin,
        model_name: format!(
            "{}_histosonics_like_256_element_full_patient_skin_arc",
            anatomy.label()
        ),
    })
}

fn abdominal_arc_points(
    count: usize,
    focus: Point3,
    radius: f64,
    half_angle: f64,
    cutout: f64,
) -> Vec<Point3> {
    let left = count / 2;
    let right = count - left;
    let mut points = Vec::with_capacity(count);
    for idx in 0..left {
        let t = if left > 1 {
            idx as f64 / (left - 1) as f64
        } else {
            0.0
        };
        points.push(abdominal_arc_point(focus, radius, -half_angle, -cutout, t));
    }
    for idx in 0..right {
        let t = if right > 1 {
            idx as f64 / (right - 1) as f64
        } else {
            0.0
        };
        points.push(abdominal_arc_point(focus, radius, cutout, half_angle, t));
    }
    points
}

fn abdominal_arc_point(focus: Point3, radius: f64, a: f64, b: f64, t: f64) -> Point3 {
    let theta = a + (b - a) * t;
    Point3 {
        x_m: focus.x_m - radius * theta.cos(),
        y_m: focus.y_m + radius * theta.sin(),
        z_m: focus.z_m,
    }
}

fn abdominal_imaging_points(skin: Point3, aperture_m: f64, count: usize) -> Vec<Point3> {
    (0..count)
        .map(|idx| {
            let t = if count > 1 {
                idx as f64 / (count - 1) as f64
            } else {
                0.5
            };
            Point3 {
                x_m: skin.x_m - ABDOMINAL_SKIN_CLEARANCE_M,
                y_m: skin.y_m + (t - 0.5) * aperture_m,
                z_m: skin.z_m,
            }
        })
        .collect()
}

fn connected_body_component(
    tissue: &Array2<bool>,
    target: &Array2<bool>,
) -> KwaversResult<Array2<bool>> {
    let seed = centroid_index(target).ok_or_else(|| {
        KwaversError::InvalidInput("connected body component requires a target seed".to_owned())
    })?;
    let (nx, ny) = tissue.dim();
    let mut body = Array2::<bool>::from_elem((nx, ny), false);
    let mut queue = VecDeque::from([seed]);
    while let Some((ix, iy)) = queue.pop_front() {
        if ix >= nx || iy >= ny || body[[ix, iy]] || !tissue[[ix, iy]] {
            continue;
        }
        body[[ix, iy]] = true;
        if ix > 0 {
            queue.push_back((ix - 1, iy));
        }
        if iy > 0 {
            queue.push_back((ix, iy - 1));
        }
        if ix + 1 < nx {
            queue.push_back((ix + 1, iy));
        }
        if iy + 1 < ny {
            queue.push_back((ix, iy + 1));
        }
    }
    if body.iter().filter(|active| **active).count() < 16 {
        return Err(KwaversError::InvalidInput(
            "connected abdominal body support is too small".to_owned(),
        ));
    }
    Ok(body)
}

fn left_skin_contact(
    body: &Array2<bool>,
    spacing_x_m: f64,
    spacing_y_m: f64,
    focus: Point3,
) -> KwaversResult<Point3> {
    let (nx, ny) = body.dim();
    let center = centered_origin_2d(nx, ny);
    let iy = ((focus.y_m / spacing_y_m) + center.1)
        .round()
        .clamp(0.0, ny.saturating_sub(1) as f64) as usize;
    for ix in 0..nx {
        if body[[ix, iy]] {
            return Ok(Point3 {
                x_m: (ix as f64 - center.0) * spacing_x_m,
                y_m: (iy as f64 - center.1) * spacing_y_m,
                z_m: 0.0,
            });
        }
    }
    let boundary = boundary_points_2d(body, spacing_x_m, spacing_y_m, 1);
    boundary
        .into_iter()
        .min_by(|a, b| distance_2d(*a, focus).total_cmp(&distance_2d(*b, focus)))
        .ok_or_else(|| KwaversError::InvalidInput("abdominal body boundary is empty".to_owned()))
}
