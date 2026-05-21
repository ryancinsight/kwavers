//! Full-volume transcranial placement context.

use std::f64::consts::PI;

use crate::{
    core::error::{KwaversError, KwaversResult},
    domain::source::transducers::focused::{BowlConfig, BowlTransducer},
};
use ndarray::{s, Array2, Array3};

use super::super::scene::target_index_from_mask_fraction_3d;
use super::super::TheranosticInverseConfig;
use super::surface::{nearest_surface_point, surface_points_3d};
use super::{distance_3d, validate_spacing, volume_bbox, volume_center, PlacementContext, Point3};

const FOCUSED_BOWL_POLAR_SPAN_RAD: f64 = 0.58 * PI;

pub fn build_brain_placement_context(
    ct_volume_hu: &Array3<f64>,
    spacing_mm: [f64; 3],
    slice_index: usize,
    config: &TheranosticInverseConfig,
    target_fraction_xyz: Option<[f64; 3]>,
) -> KwaversResult<PlacementContext> {
    validate_spacing(spacing_mm)?;
    let (nx, ny, nz) = ct_volume_hu.dim();
    if slice_index >= nz {
        return Err(KwaversError::InvalidInput(format!(
            "brain placement slice {slice_index} out of bounds for {nz} slices"
        )));
    }

    let sx = spacing_mm[0] * 1.0e-3;
    let sy = spacing_mm[1] * 1.0e-3;
    let sz = spacing_mm[2] * 1.0e-3;
    let body = ct_volume_hu.mapv(|hu| hu > -300.0);
    let brain = ct_volume_hu.mapv(|hu| hu > -300.0 && hu < 300.0);
    let target_support = if brain.iter().any(|active| *active) {
        &brain
    } else {
        &body
    };
    let bbox = volume_bbox(&body)?;
    let areas = axial_areas(&body);
    let peak_z = areas
        .iter()
        .enumerate()
        .max_by_key(|(_, area)| **area)
        .map(|(idx, _)| idx)
        .unwrap_or(slice_index);
    let superior_positive = areas[nz - 1] < areas[0];
    let calvarium_range = if superior_positive {
        peak_z..=bbox.z1
    } else {
        bbox.z0..=peak_z
    };
    let center = if let Some(fraction) = target_fraction_xyz {
        let target_index = target_index_from_mask_fraction_3d(target_support, fraction)?;
        point_from_index(target_index, nx, ny, nz, sx, sy, sz)
    } else {
        centroid_3d(&body, sx, sy, sz, Some(calvarium_range.clone()))
            .unwrap_or_else(|| volume_center(nx, ny, nz, sx, sy, sz))
    };
    let radius = calvarium_radius(&body, sx, sy, sz, center, calvarium_range.clone()) + 0.015;
    let frequency_hz = config.frequencies_hz.first().copied().ok_or_else(|| {
        KwaversError::InvalidInput(
            "brain placement context requires at least one source frequency".to_owned(),
        )
    })?;
    let therapy_points_m = focused_bowl_cap_points(
        config.element_count,
        center,
        radius.max(config.focal_radius_m),
        superior_positive,
        frequency_hz,
        config.source_pressure_pa,
    )?;
    let surface_stride = (nx.max(ny).max(nz) / 96).clamp(1, 8);
    let body_surface_points_m =
        surface_points_3d(&body, sx, sy, sz, Some(calvarium_range), surface_stride);

    let slice = ct_volume_hu.slice(s![.., .., slice_index]).to_owned();
    let body_slice = body.slice(s![.., .., slice_index]).to_owned();
    let target_mask = synthetic_focus_mask(&body_slice, sx.max(sy), (center.x_m, center.y_m));
    let skin_contact_m = nearest_surface_point(&body_surface_points_m, center).unwrap_or(center);

    Ok(PlacementContext {
        ct_hu: slice,
        body_mask: body_slice,
        target_mask,
        spacing_x_m: sx,
        spacing_y_m: sy,
        slice_index,
        therapy_points_m,
        imaging_points_m: Vec::new(),
        body_surface_points_m,
        focus_m: center,
        skin_contact_m,
        model_name: "focused_bowl_major_cap_3d".to_owned(),
    })
}

fn focused_bowl_cap_points(
    count: usize,
    center: Point3,
    radius: f64,
    superior_positive: bool,
    frequency_hz: f64,
    amplitude_pa: f64,
) -> KwaversResult<Vec<Point3>> {
    let z_sign = if superior_positive { 1.0 } else { -1.0 };
    let vertex_m = [center.x_m, center.y_m, center.z_m + z_sign * radius];
    let focus_m = [center.x_m, center.y_m, center.z_m];
    let config =
        BowlConfig::from_vertex_focus(vertex_m, focus_m, 2.0 * radius, frequency_hz, amplitude_pa);
    let bowl = BowlTransducer::with_polar_span(config, FOCUSED_BOWL_POLAR_SPAN_RAD, count)?;

    Ok(bowl
        .element_positions()
        .iter()
        .map(|position| Point3 {
            x_m: position[0],
            y_m: position[1],
            z_m: position[2],
        })
        .collect())
}

fn point_from_index(
    index: [usize; 3],
    nx: usize,
    ny: usize,
    nz: usize,
    sx: f64,
    sy: f64,
    sz: f64,
) -> Point3 {
    Point3 {
        x_m: (index[0] as f64 - (nx - 1) as f64 * 0.5) * sx,
        y_m: (index[1] as f64 - (ny - 1) as f64 * 0.5) * sy,
        z_m: (index[2] as f64 - (nz - 1) as f64 * 0.5) * sz,
    }
}

fn synthetic_focus_mask(body: &Array2<bool>, spacing_m: f64, focus: (f64, f64)) -> Array2<bool> {
    let (nx, ny) = body.dim();
    let cx = (nx - 1) as f64 * 0.5 + focus.0 / spacing_m;
    let cy = (ny - 1) as f64 * 0.5 + focus.1 / spacing_m;
    let rx = 6.0e-3 / spacing_m;
    let ry = 8.0e-3 / spacing_m;
    Array2::from_shape_fn((nx, ny), |(ix, iy)| {
        body[[ix, iy]] && ((ix as f64 - cx) / rx).powi(2) + ((iy as f64 - cy) / ry).powi(2) <= 1.0
    })
}

fn axial_areas(mask: &Array3<bool>) -> Vec<usize> {
    let (_, _, nz) = mask.dim();
    (0..nz)
        .map(|iz| {
            mask.slice(s![.., .., iz])
                .iter()
                .filter(|active| **active)
                .count()
        })
        .collect()
}

fn centroid_3d(
    mask: &Array3<bool>,
    sx: f64,
    sy: f64,
    sz: f64,
    z_range: Option<std::ops::RangeInclusive<usize>>,
) -> Option<Point3> {
    let (nx, ny, nz) = mask.dim();
    let mut sum = Point3 {
        x_m: 0.0,
        y_m: 0.0,
        z_m: 0.0,
    };
    let mut count = 0.0;
    for ((ix, iy, iz), active) in mask.indexed_iter() {
        if *active && z_range.as_ref().is_none_or(|range| range.contains(&iz)) {
            sum.x_m += (ix as f64 - (nx - 1) as f64 * 0.5) * sx;
            sum.y_m += (iy as f64 - (ny - 1) as f64 * 0.5) * sy;
            sum.z_m += (iz as f64 - (nz - 1) as f64 * 0.5) * sz;
            count += 1.0;
        }
    }
    (count > 0.0).then_some(Point3 {
        x_m: sum.x_m / count,
        y_m: sum.y_m / count,
        z_m: sum.z_m / count,
    })
}

fn calvarium_radius(
    mask: &Array3<bool>,
    sx: f64,
    sy: f64,
    sz: f64,
    center: Point3,
    z_range: std::ops::RangeInclusive<usize>,
) -> f64 {
    let (nx, ny, nz) = mask.dim();
    mask.indexed_iter()
        .filter_map(|((ix, iy, iz), active)| {
            if *active && z_range.contains(&iz) {
                let point = Point3 {
                    x_m: (ix as f64 - (nx - 1) as f64 * 0.5) * sx,
                    y_m: (iy as f64 - (ny - 1) as f64 * 0.5) * sy,
                    z_m: (iz as f64 - (nz - 1) as f64 * 0.5) * sz,
                };
                Some(distance_3d(point, center))
            } else {
                None
            }
        })
        .fold(0.0, f64::max)
}
