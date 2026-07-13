//! Bounding-box geometry for nonlinear 3-D CT volume cropping.

use std::collections::VecDeque;

use leto::{Array2, Array3};

use crate::therapy::theranostic_guidance::geometry::IndexBounds3;
use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::super::skin::nearest_external_skin_point;
use super::super::super::AnatomyKind;
use super::super::aperture_bowl::abdominal_bowl_radius_m;
use super::centroid::centroid_float;

const TARGET_TO_WINDOW_MARGIN_M: f64 = 0.012;

pub(super) fn crop_bbox(
    anatomy: AnatomyKind,
    body: &Array3<bool>,
    target: Option<&Array3<bool>>,
    aperture_skin_index: Option<[f64; 3]>,
    target_center_index: Option<[f64; 3]>,
    spacing_mm: [f64; 3],
    treatment_window_radius_m: f64,
) -> KwaversResult<IndexBounds3> {
    match anatomy {
        AnatomyKind::Brain => brain_cube_bbox(
            body,
            target_center_index,
            spacing_mm,
            treatment_window_radius_m,
        ),
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            let target = target.ok_or_else(|| {
                KwaversError::InvalidInput("abdominal target mask is required".to_owned())
            })?;
            path_cube_bbox(
                body,
                target,
                aperture_skin_index,
                spacing_mm,
                treatment_window_radius_m,
            )
        }
    }
}

fn brain_cube_bbox(
    body: &Array3<bool>,
    target_center_index: Option<[f64; 3]>,
    spacing_mm: [f64; 3],
    treatment_window_radius_m: f64,
) -> KwaversResult<IndexBounds3> {
    if treatment_window_radius_m > 0.0 {
        let center = target_center_index.ok_or_else(|| {
            KwaversError::InvalidInput("brain treatment-window target index is required".to_owned())
        })?;
        return Ok(cube_from_center_radius(
            body.shape(),
            center,
            treatment_window_radius_m,
            spacing_mm,
        ));
    }
    body_cube_bbox(body, spacing_mm)
}

fn body_cube_bbox(body: &Array3<bool>, spacing_mm: [f64; 3]) -> KwaversResult<IndexBounds3> {
    let bounds = mask_bounds(body)?;
    let center = [
        0.5 * (bounds.x0 + bounds.x1) as f64,
        0.5 * (bounds.y0 + bounds.y1) as f64,
        0.5 * (bounds.z0 + bounds.z1) as f64,
    ];
    let radius_m = [
        0.5 * (bounds.x1 - bounds.x0 + 1) as f64 * spacing_mm[0] * 1.0e-3,
        0.5 * (bounds.y1 - bounds.y0 + 1) as f64 * spacing_mm[1] * 1.0e-3,
        0.5 * (bounds.z1 - bounds.z0 + 1) as f64 * spacing_mm[2] * 1.0e-3,
    ]
    .into_iter()
    .fold(0.0, f64::max)
        + 0.01;
    Ok(cube_from_center_radius(
        body.shape(),
        center,
        radius_m,
        spacing_mm,
    ))
}

fn path_cube_bbox(
    body: &Array3<bool>,
    target: &Array3<bool>,
    aperture_skin_index: Option<[f64; 3]>,
    spacing_mm: [f64; 3],
    treatment_window_radius_m: f64,
) -> KwaversResult<IndexBounds3> {
    let focus = centroid_float(target, None).ok_or_else(|| {
        KwaversError::InvalidInput("abdominal nonlinear target mask is empty".to_owned())
    })?;
    let target_bounds = mask_bounds(target)?;
    let skin = aperture_skin_index.ok_or_else(|| {
        KwaversError::InvalidInput("abdominal aperture skin index is required".to_owned())
    })?;
    let center = [
        0.5 * (focus[0] + skin[0]),
        0.5 * (focus[1] + skin[1]),
        0.5 * (focus[2] + skin[2]),
    ];
    let target_radius_from_focus_m = max_distance_to_bbox(focus, target_bounds, spacing_mm);
    let target_radius_from_center_m = max_distance_to_bbox(center, target_bounds, spacing_mm);
    let skin_distance_m = physical_distance(focus, skin, spacing_mm);
    let bowl_radius_m = abdominal_bowl_radius_m(skin_distance_m);
    let bowl_radius_from_center_m =
        (bowl_radius_m - 0.5 * skin_distance_m).max(0.0) + TARGET_TO_WINDOW_MARGIN_M;
    let focus_margin_m = treatment_window_radius_m
        .max(target_radius_from_focus_m + TARGET_TO_WINDOW_MARGIN_M)
        .max(0.025);
    let radius_m = (0.5 * skin_distance_m + focus_margin_m)
        .max(target_radius_from_center_m + TARGET_TO_WINDOW_MARGIN_M)
        .max(bowl_radius_from_center_m);
    Ok(cube_from_center_radius(
        body.shape(),
        center,
        radius_m,
        spacing_mm,
    ))
}

pub(super) fn planned_abdominal_skin_index(
    body: &Array3<bool>,
    target: &Array3<bool>,
    spacing_mm: [f64; 3],
) -> KwaversResult<[f64; 3]> {
    let z = largest_target_slice(target)?;
    let [nx, ny, _] = target.shape();
    let body_slice = Array2::from_shape_fn((nx, ny), |[ix, iy]| body[[ix, iy, z]]);
    let target_slice = Array2::from_shape_fn((nx, ny), |[ix, iy]| target[[ix, iy, z]]);
    let body_component = connected_body_component(&body_slice, &target_slice)?;
    let focus = centroid_2d(&target_slice, spacing_mm)?;
    let sx = spacing_mm[0] * 1.0e-3;
    let sy = spacing_mm[1] * 1.0e-3;
    let skin_m = nearest_external_skin_point(&body_component, sx, sy, focus.0, focus.1)?;
    let center = (0.5 * (nx - 1) as f64, 0.5 * (ny - 1) as f64);
    Ok([
        skin_m.x_m / sx + center.0,
        skin_m.y_m / sy + center.1,
        z as f64,
    ])
}

fn largest_target_slice(target: &Array3<bool>) -> KwaversResult<usize> {
    let [_, _, nz] = target.shape();
    let mut best = None;
    for z in 0..nz {
        let count = (0..target.shape()[0])
            .flat_map(|x| (0..target.shape()[1]).map(move |y| (x, y)))
            .filter(|(x, y)| target[[*x, *y, z]])
            .count();
        if count > best.map_or(0, |(_, current)| current) {
            best = Some((z, count));
        }
    }
    best.filter(|(_, count)| *count > 0)
        .map(|(z, _)| z)
        .ok_or_else(|| {
            KwaversError::InvalidInput("abdominal target has no active slice".to_owned())
        })
}

fn centroid_2d(target: &Array2<bool>, spacing_mm: [f64; 3]) -> KwaversResult<(f64, f64)> {
    let [nx, ny] = target.shape();
    let center = (0.5 * (nx - 1) as f64, 0.5 * (ny - 1) as f64);
    let sx = spacing_mm[0] * 1.0e-3;
    let sy = spacing_mm[1] * 1.0e-3;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0.0;
    for ([ix, iy], active) in target.indexed_iter() {
        if *active {
            sum_x += (ix as f64 - center.0) * sx;
            sum_y += (iy as f64 - center.1) * sy;
            count += 1.0;
        }
    }
    if count > 0.0 {
        Ok((sum_x / count, sum_y / count))
    } else {
        Err(KwaversError::InvalidInput(
            "abdominal target slice is empty".to_owned(),
        ))
    }
}

fn connected_body_component(
    body: &Array2<bool>,
    target: &Array2<bool>,
) -> KwaversResult<Array2<bool>> {
    let seed = target
        .indexed_iter()
        .find_map(|(idx, active)| active.then_some((idx[0], idx[1])))
        .ok_or_else(|| KwaversError::InvalidInput("abdominal target slice is empty".to_owned()))?;
    let [nx, ny] = body.shape();
    let mut component = Array2::<bool>::from_elem((nx, ny), false);
    let mut queue = VecDeque::from([seed]);
    while let Some((ix, iy)) = queue.pop_front() {
        if ix >= nx || iy >= ny || component[[ix, iy]] || !body[[ix, iy]] {
            continue;
        }
        component[[ix, iy]] = true;
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
    if component.iter().filter(|active| **active).count() < 16 {
        return Err(KwaversError::InvalidInput(
            "abdominal body component is too small".to_owned(),
        ));
    }
    Ok(component)
}

fn cube_from_center_radius(
    dims: [usize; 3],
    center: [f64; 3],
    radius_m: f64,
    spacing_mm: [f64; 3],
) -> IndexBounds3 {
    let rx = (radius_m / (spacing_mm[0] * 1.0e-3)).ceil() as usize;
    let ry = (radius_m / (spacing_mm[1] * 1.0e-3)).ceil() as usize;
    let rz = (radius_m / (spacing_mm[2] * 1.0e-3)).ceil() as usize;
    IndexBounds3 {
        x0: (center[0].round() as isize - rx as isize).max(0) as usize,
        x1: (center[0].round() as usize + rx).min(dims[0] - 1),
        y0: (center[1].round() as isize - ry as isize).max(0) as usize,
        y1: (center[1].round() as usize + ry).min(dims[1] - 1),
        z0: (center[2].round() as isize - rz as isize).max(0) as usize,
        z1: (center[2].round() as usize + rz).min(dims[2] - 1),
    }
}

fn mask_bounds(mask: &Array3<bool>) -> KwaversResult<IndexBounds3> {
    let mut b = IndexBounds3 {
        x0: usize::MAX,
        x1: 0,
        y0: usize::MAX,
        y1: 0,
        z0: usize::MAX,
        z1: 0,
    };
    let mut any = false;
    for ([ix, iy, iz], active) in mask.indexed_iter() {
        if *active {
            b.x0 = b.x0.min(ix);
            b.x1 = b.x1.max(ix);
            b.y0 = b.y0.min(iy);
            b.y1 = b.y1.max(iy);
            b.z0 = b.z0.min(iz);
            b.z1 = b.z1.max(iz);
            any = true;
        }
    }
    any.then_some(b)
        .ok_or_else(|| KwaversError::InvalidInput("3-D mask support is empty".to_owned()))
}

fn max_distance_to_bbox(center: [f64; 3], bbox: IndexBounds3, spacing_mm: [f64; 3]) -> f64 {
    let mut max_distance: f64 = 0.0;
    for x in [bbox.x0, bbox.x1] {
        for y in [bbox.y0, bbox.y1] {
            for z in [bbox.z0, bbox.z1] {
                max_distance = max_distance.max(physical_distance(
                    center,
                    [x as f64, y as f64, z as f64],
                    spacing_mm,
                ));
            }
        }
    }
    max_distance
}

pub(super) fn physical_distance(a: [f64; 3], b: [f64; 3], spacing_mm: [f64; 3]) -> f64 {
    let dx = (a[0] - b[0]) * spacing_mm[0] * 1.0e-3;
    let dy = (a[1] - b[1]) * spacing_mm[1] * 1.0e-3;
    let dz = (a[2] - b[2]) * spacing_mm[2] * 1.0e-3;
    dx.hypot(dy).hypot(dz)
}

#[cfg(test)]
mod tests {
    use leto::Array3;

    use super::*;

    #[test]
    fn abdominal_path_crop_contains_focused_bowl_standoff() {
        let n = 240;
        let spacing_mm = [1.0, 1.0, 1.0];
        let focus = [120_usize, 40_usize, 120_usize];
        let skin = [120.0, 140.0, 120.0];
        let mut body = Array3::from_elem((n, n, n), false);
        for x in 40..200 {
            for y in 20..=140 {
                for z in 40..200 {
                    body[[x, y, z]] = true;
                }
            }
        }
        let mut target = Array3::from_elem((n, n, n), false);
        target[[focus[0], focus[1], focus[2]]] = true;

        let bbox = path_cube_bbox(&body, &target, Some(skin), spacing_mm, 0.04).unwrap();
        let depth_m = physical_distance(
            [focus[0] as f64, focus[1] as f64, focus[2] as f64],
            skin,
            spacing_mm,
        );
        let source_y = 0.5 * (focus[1] as f64 + skin[1])
            + (abdominal_bowl_radius_m(depth_m) - 0.5 * depth_m) / (spacing_mm[1] * 1.0e-3);

        assert!(
            bbox.y1 as f64 >= source_y,
            "crop y1={} must include focused bowl source vertex at y={source_y:.3}",
            bbox.y1
        );
    }
}
