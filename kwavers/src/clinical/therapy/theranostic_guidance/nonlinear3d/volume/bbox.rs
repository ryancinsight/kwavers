//! Bounding-box geometry for nonlinear 3-D CT volume cropping.

use ndarray::Array3;

use crate::clinical::therapy::theranostic_guidance::geometry::IndexBounds3;
use crate::core::error::{KwaversError, KwaversResult};

use super::super::super::AnatomyKind;
use super::centroid::centroid_float;
use super::mask::nearest_boundary;

pub(super) fn crop_bbox(
    anatomy: AnatomyKind,
    body: &Array3<bool>,
    target: Option<&Array3<bool>>,
    spacing_mm: [f64; 3],
) -> KwaversResult<IndexBounds3> {
    match anatomy {
        AnatomyKind::Brain => body_cube_bbox(body, spacing_mm),
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            let target = target.ok_or_else(|| {
                KwaversError::InvalidInput("abdominal target mask is required".to_owned())
            })?;
            path_cube_bbox(body, target, spacing_mm)
        }
    }
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
        body.dim(),
        center,
        radius_m,
        spacing_mm,
    ))
}

fn path_cube_bbox(
    body: &Array3<bool>,
    target: &Array3<bool>,
    spacing_mm: [f64; 3],
) -> KwaversResult<IndexBounds3> {
    let focus = centroid_float(target).ok_or_else(|| {
        KwaversError::InvalidInput("abdominal nonlinear target mask is empty".to_owned())
    })?;
    let skin = nearest_boundary(body, focus, spacing_mm)?;
    let center = [
        0.5 * (focus[0] + skin[0]),
        0.5 * (focus[1] + skin[1]),
        0.5 * (focus[2] + skin[2]),
    ];
    let target_bounds = mask_bounds(target)?;
    let target_radius_m = max_distance_to_bbox(center, target_bounds, spacing_mm);
    let skin_distance_m = physical_distance(focus, skin, spacing_mm);
    let radius_m = target_radius_m.max(0.55 * skin_distance_m) + 0.025;
    Ok(cube_from_center_radius(
        body.dim(),
        center,
        radius_m,
        spacing_mm,
    ))
}

fn cube_from_center_radius(
    dims: (usize, usize, usize),
    center: [f64; 3],
    radius_m: f64,
    spacing_mm: [f64; 3],
) -> IndexBounds3 {
    let rx = (radius_m / (spacing_mm[0] * 1.0e-3)).ceil() as usize;
    let ry = (radius_m / (spacing_mm[1] * 1.0e-3)).ceil() as usize;
    let rz = (radius_m / (spacing_mm[2] * 1.0e-3)).ceil() as usize;
    IndexBounds3 {
        x0: (center[0].round() as isize - rx as isize).max(0) as usize,
        x1: (center[0].round() as usize + rx).min(dims.0 - 1),
        y0: (center[1].round() as isize - ry as isize).max(0) as usize,
        y1: (center[1].round() as usize + ry).min(dims.1 - 1),
        z0: (center[2].round() as isize - rz as isize).max(0) as usize,
        z1: (center[2].round() as usize + rz).min(dims.2 - 1),
    }
}

fn mask_bounds(mask: &Array3<bool>) -> KwaversResult<IndexBounds3> {
    let mut b = IndexBounds3 {
        x0: usize::MAX, x1: 0,
        y0: usize::MAX, y1: 0,
        z0: usize::MAX, z1: 0,
    };
    let mut any = false;
    for ((ix, iy, iz), active) in mask.indexed_iter() {
        if *active {
            b.x0 = b.x0.min(ix); b.x1 = b.x1.max(ix);
            b.y0 = b.y0.min(iy); b.y1 = b.y1.max(iy);
            b.z0 = b.z0.min(iz); b.z1 = b.z1.max(iz);
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
