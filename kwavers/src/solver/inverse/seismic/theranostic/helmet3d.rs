//! CT-derived 3-D helmet placement for visual verification.

use ndarray::{s, Array3};

use crate::core::error::{KwaversError, KwaversResult};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3 {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

#[derive(Clone, Debug)]
pub struct BrainHelmetPlacement3D {
    pub head_surface_points_m: Vec<Point3>,
    pub skull_surface_points_m: Vec<Point3>,
    pub therapy_elements_m: Vec<Point3>,
    pub beam_start_points_m: Vec<Point3>,
    pub beam_end_points_m: Vec<Point3>,
    pub skull_intersections_m: Vec<Point3>,
    pub focus_m: Point3,
    pub helmet_radius_m: f64,
    pub intersection_fraction: f64,
}

pub fn plan_brain_helmet_placement(
    ct_hu: &Array3<f64>,
    spacing_mm: [f64; 3],
    element_count: usize,
    surface_stride: usize,
    body_hu_threshold: f64,
    skull_hu_threshold: f64,
) -> KwaversResult<BrainHelmetPlacement3D> {
    if element_count < 16 {
        return Err(KwaversError::InvalidInput(
            "3-D helmet placement requires at least 16 elements".to_owned(),
        ));
    }
    if spacing_mm
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(KwaversError::InvalidInput(
            "3-D helmet placement requires positive finite CT spacing".to_owned(),
        ));
    }
    let body_mask = ct_hu.mapv(|hu| hu.is_finite() && hu >= body_hu_threshold);
    let skull_mask = ct_hu.mapv(|hu| hu.is_finite() && hu >= skull_hu_threshold);
    let bounds = body_bounds(&body_mask)?;
    let areas = axial_areas(&body_mask);
    let peak_z = areas
        .iter()
        .enumerate()
        .max_by_key(|(_, area)| **area)
        .map(|(idx, _)| idx)
        .unwrap_or((bounds.min[2] + bounds.max[2]) / 2);
    let superior_positive =
        areas.last().copied().unwrap_or(0) < areas.first().copied().unwrap_or(0);
    let calvarium_min_z = if superior_positive {
        peak_z
    } else {
        bounds.min[2]
    };
    let calvarium_max_z = if superior_positive {
        bounds.max[2]
    } else {
        peak_z
    };
    let center_index = centroid_index_in_z_range(&body_mask, calvarium_min_z, calvarium_max_z)
        .unwrap_or([
            0.5 * (bounds.min[0] + bounds.max[0]) as f64,
            0.5 * (bounds.min[1] + bounds.max[1]) as f64,
            0.5 * (calvarium_min_z + calvarium_max_z) as f64,
        ]);
    let spacing_m = [
        spacing_mm[0] * 1.0e-3,
        spacing_mm[1] * 1.0e-3,
        spacing_mm[2] * 1.0e-3,
    ];
    let focus = Point3 {
        x_m: 0.0,
        y_m: 0.0,
        z_m: 0.0,
    };
    let stride = surface_stride.max(1);
    let head_surface_points_m = surface_points(
        &body_mask,
        spacing_m,
        center_index,
        stride,
        calvarium_min_z,
        calvarium_max_z,
    );
    let skull_surface_points_m = surface_points(
        &skull_mask,
        spacing_m,
        center_index,
        stride,
        calvarium_min_z,
        calvarium_max_z,
    );
    let head_radius = body_radius(
        &body_mask,
        spacing_m,
        center_index,
        calvarium_min_z,
        calvarium_max_z,
    );
    let helmet_radius_m = (head_radius + 0.015).max(0.150);
    let therapy_elements_m =
        calvarium_cap_elements(element_count, helmet_radius_m, focus, superior_positive);
    let (beam_start_points_m, beam_end_points_m, skull_intersections_m) = sample_beams(
        &therapy_elements_m,
        focus,
        &skull_mask,
        spacing_m,
        center_index,
    );
    let intersection_fraction = if beam_start_points_m.is_empty() {
        0.0
    } else {
        skull_intersections_m.len() as f64 / beam_start_points_m.len() as f64
    };

    Ok(BrainHelmetPlacement3D {
        head_surface_points_m,
        skull_surface_points_m,
        therapy_elements_m,
        beam_start_points_m,
        beam_end_points_m,
        skull_intersections_m,
        focus_m: focus,
        helmet_radius_m,
        intersection_fraction,
    })
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min: [usize; 3],
    max: [usize; 3],
}

fn body_bounds(mask: &Array3<bool>) -> KwaversResult<Bounds> {
    let mut min = [usize::MAX; 3];
    let mut max = [0usize; 3];
    let mut any = false;
    for ((ix, iy, iz), active) in mask.indexed_iter() {
        if *active {
            min[0] = min[0].min(ix);
            min[1] = min[1].min(iy);
            min[2] = min[2].min(iz);
            max[0] = max[0].max(ix);
            max[1] = max[1].max(iy);
            max[2] = max[2].max(iz);
            any = true;
        }
    }
    if any {
        Ok(Bounds { min, max })
    } else {
        Err(KwaversError::InvalidInput(
            "3-D helmet placement found no CT head support voxels".to_owned(),
        ))
    }
}

fn surface_points(
    mask: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
    stride: usize,
    min_z: usize,
    max_z: usize,
) -> Vec<Point3> {
    mask.indexed_iter()
        .filter_map(|((ix, iy, iz), active)| {
            (*active
                && iz >= min_z
                && iz <= max_z
                && is_surface(mask, ix, iy, iz)
                && (ix + iy + iz) % stride == 0)
                .then(|| voxel_to_point(ix, iy, iz, spacing_m, center_index))
        })
        .collect()
}

fn body_radius(
    mask: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
    min_z: usize,
    max_z: usize,
) -> f64 {
    mask.indexed_iter()
        .filter_map(|((ix, iy, iz), active)| {
            (*active && iz >= min_z && iz <= max_z).then(|| {
                let point = voxel_to_point(ix, iy, iz, spacing_m, center_index);
                point.x_m.hypot(point.y_m).hypot(point.z_m)
            })
        })
        .fold(0.0, f64::max)
}

fn calvarium_cap_elements(
    count: usize,
    radius_m: f64,
    focus: Point3,
    superior_positive: bool,
) -> Vec<Point3> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let z_min = -0.28;
    let z_max = 0.98;
    let sign = if superior_positive { 1.0 } else { -1.0 };
    (0..count)
        .map(|idx| {
            let t = (idx as f64 + 0.5) / count as f64;
            let z_unit = z_min + (z_max - z_min) * t;
            let radial = (1.0 - z_unit * z_unit).sqrt();
            let phi = idx as f64 * golden;
            Point3 {
                x_m: focus.x_m + radius_m * radial * phi.cos(),
                y_m: focus.y_m + radius_m * radial * phi.sin(),
                z_m: focus.z_m + sign * radius_m * z_unit,
            }
        })
        .collect()
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

fn centroid_index_in_z_range(mask: &Array3<bool>, min_z: usize, max_z: usize) -> Option<[f64; 3]> {
    let mut sum = [0.0; 3];
    let mut count = 0.0;
    for ((ix, iy, iz), active) in mask.indexed_iter() {
        if *active && iz >= min_z && iz <= max_z {
            sum[0] += ix as f64;
            sum[1] += iy as f64;
            sum[2] += iz as f64;
            count += 1.0;
        }
    }
    (count > 0.0).then_some([sum[0] / count, sum[1] / count, sum[2] / count])
}

fn sample_beams(
    elements: &[Point3],
    focus: Point3,
    skull_mask: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
) -> (Vec<Point3>, Vec<Point3>, Vec<Point3>) {
    let beam_count = 72usize.min(elements.len()).max(1);
    let step = (elements.len() / beam_count).max(1);
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut intersections = Vec::new();
    for element in elements.iter().step_by(step).take(beam_count) {
        starts.push(*element);
        ends.push(focus);
        if let Some(point) =
            first_skull_intersection(*element, focus, skull_mask, spacing_m, center_index)
        {
            intersections.push(point);
        }
    }
    (starts, ends, intersections)
}

fn first_skull_intersection(
    start: Point3,
    end: Point3,
    skull_mask: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
) -> Option<Point3> {
    let steps = 320usize;
    for step in 0..=steps {
        let t = step as f64 / steps as f64;
        let point = Point3 {
            x_m: start.x_m + t * (end.x_m - start.x_m),
            y_m: start.y_m + t * (end.y_m - start.y_m),
            z_m: start.z_m + t * (end.z_m - start.z_m),
        };
        if point_in_mask(point, skull_mask, spacing_m, center_index) {
            return Some(point);
        }
    }
    None
}

fn point_in_mask(
    point: Point3,
    mask: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
) -> bool {
    let ix = (point.x_m / spacing_m[0] + center_index[0]).round() as isize;
    let iy = (point.y_m / spacing_m[1] + center_index[1]).round() as isize;
    let iz = (point.z_m / spacing_m[2] + center_index[2]).round() as isize;
    let shape = mask.dim();
    if ix < 0 || iy < 0 || iz < 0 {
        return false;
    }
    let (ix, iy, iz) = (ix as usize, iy as usize, iz as usize);
    ix < shape.0 && iy < shape.1 && iz < shape.2 && mask[[ix, iy, iz]]
}

fn is_surface(mask: &Array3<bool>, ix: usize, iy: usize, iz: usize) -> bool {
    let (nx, ny, nz) = mask.dim();
    ix == 0
        || iy == 0
        || iz == 0
        || ix + 1 == nx
        || iy + 1 == ny
        || iz + 1 == nz
        || !mask[[ix - 1, iy, iz]]
        || !mask[[ix + 1, iy, iz]]
        || !mask[[ix, iy - 1, iz]]
        || !mask[[ix, iy + 1, iz]]
        || !mask[[ix, iy, iz - 1]]
        || !mask[[ix, iy, iz + 1]]
}

fn voxel_to_point(
    ix: usize,
    iy: usize,
    iz: usize,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
) -> Point3 {
    Point3 {
        x_m: (ix as f64 - center_index[0]) * spacing_m[0],
        y_m: (iy as f64 - center_index[1]) * spacing_m[1],
        z_m: (iz as f64 - center_index[2]) * spacing_m[2],
    }
}
