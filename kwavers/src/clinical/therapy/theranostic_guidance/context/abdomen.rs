//! Full-slice abdominal placement context.

use std::collections::VecDeque;

use crate::core::constants::fundamental::HU_ABDOMEN_BODY_THRESHOLD;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2, Array3};

use super::super::aperture::{
    abdominal_aperture_frame, abdominal_arc_point_2d, abdominal_arc_spec,
    abdominal_imaging_point_2d, AbdominalApertureFrame,
};
use super::super::skin::nearest_external_skin_point;
use super::super::{
    medium::{largest_connected_target_component, largest_target_slice},
    AnatomyKind, TheranosticInverseConfig,
};
use super::surface::surface_points_2d;
use super::{centroid_2d, centroid_index, validate_spacing, PlacementContext, Point3};

pub fn build_abdominal_placement_context(
    anatomy: AnatomyKind,
    ct_volume_hu: &Array3<f64>,
    label_volume: &Array3<i16>,
    spacing_mm: [f64; 3],
    config: &TheranosticInverseConfig,
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
    let target_mask = largest_connected_target_component(&label_slice)?;
    let tissue = Array2::from_shape_fn(ct_slice.dim(), |idx| {
        ct_slice[idx] > HU_ABDOMEN_BODY_THRESHOLD || label_slice[idx] > 0
    });
    let body_mask = connected_body_component(&tissue, &target_mask)?;
    let sx = spacing_mm[0] * 1.0e-3;
    let sy = spacing_mm[1] * 1.0e-3;
    let sz = spacing_mm[2] * 1.0e-3;
    let focus_2d = centroid_2d(&target_mask, sx, sy).ok_or_else(|| {
        KwaversError::InvalidInput("abdominal placement target mask is empty".to_owned())
    })?;
    let skin_2d = nearest_external_skin_point(&body_mask, sx, sy, focus_2d.x_m, focus_2d.y_m)?;
    let skin = Point3 {
        x_m: skin_2d.x_m,
        y_m: skin_2d.y_m,
        z_m: 0.0,
    };
    let frame = abdominal_aperture_frame(focus_2d.x_m, focus_2d.y_m, skin.x_m, skin.y_m);
    let arc = abdominal_arc_spec(config, frame.depth_m);
    let focus = Point3 {
        x_m: focus_2d.x_m,
        y_m: focus_2d.y_m,
        z_m: 0.0,
    };
    let therapy_points_m = abdominal_arc_points(
        config.element_count,
        frame,
        focus,
        arc.radius_m,
        arc.half_angle_rad,
        arc.cutout_angle_rad,
    );
    let imaging_points_m = abdominal_imaging_points(frame, skin, config.central_cutout_m, 64);
    let surface_stride = (ct_slice.dim().0.max(ct_slice.dim().1) / 192).clamp(1, 6);
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
            "{}_focused_bowl_256_element_full_patient_skin_arc",
            anatomy.label()
        ),
    })
}

fn abdominal_arc_points(
    count: usize,
    frame: AbdominalApertureFrame,
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
        points.push(abdominal_arc_point(
            frame,
            focus,
            radius,
            -half_angle,
            -cutout,
            t,
        ));
    }
    for idx in 0..right {
        let t = if right > 1 {
            idx as f64 / (right - 1) as f64
        } else {
            0.0
        };
        points.push(abdominal_arc_point(
            frame, focus, radius, cutout, half_angle, t,
        ));
    }
    points
}

fn abdominal_arc_point(
    frame: AbdominalApertureFrame,
    focus: Point3,
    radius: f64,
    a: f64,
    b: f64,
    t: f64,
) -> Point3 {
    let theta = a + (b - a) * t;
    let (x_m, y_m) = abdominal_arc_point_2d(frame, focus.x_m, focus.y_m, radius, theta);
    Point3 {
        x_m,
        y_m,
        z_m: focus.z_m,
    }
}

fn abdominal_imaging_points(
    frame: AbdominalApertureFrame,
    skin: Point3,
    aperture_m: f64,
    count: usize,
) -> Vec<Point3> {
    (0..count)
        .map(|idx| {
            let t = if count > 1 {
                idx as f64 / (count - 1) as f64
            } else {
                0.5
            };
            let (x_m, y_m) = abdominal_imaging_point_2d(frame, skin.x_m, skin.y_m, aperture_m, t);
            Point3 {
                x_m,
                y_m,
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
