//! CT-derived 3-D focused-bowl placement for visual verification.
//!
//! # Theorem (Calvarium region extraction)
//!
//! For a head-only CT with voxels ordered along z:
//! Let `A(z)` be the number of body-mask voxels in axial slice `z`.
//! Define `peak_z = argmax_z A(z)`. For a typical head CT, `peak_z` falls
//! at the temporoparietal junction — the widest cranial cross-section.
//! The region `z > peak_z` (superior to the widest slice) constitutes the
//! calvarium (skull dome). Elements restricted to this z-range will cover the
//! calvarium, not the jaw or neck.
//!
//! Superior orientation is estimated by comparing the summed body-mask area in
//! the inferior third of the volume against the superior third. The end with
//! MORE body mask area is classified as inferior, because the jaw/neck
//! region has a larger cross-section than the vertex.
//!
//! # Assumptions
//!
//! - Input CT covers the head only (brain CT), not the full torso.
//! - Voxel body threshold (−350 HU) captures all head/neck tissue.
//! - The z-axis is approximately aligned with the inferior-superior axis.

use ndarray::{s, Array3};

use crate::{
    core::error::{KwaversError, KwaversResult},
    domain::source::transducers::focused::{BowlConfig, BowlTransducer},
};

use super::geometry::{is_boundary_3d, Point3};
use super::nonlinear3d::volume::centroid_float;
use super::scene::target_index_from_mask_fraction_3d;

/// Default minimum polar angle from the superior vertex [rad].
///
/// 0.22 rad ≈ 12.6° — avoids element crowding at the vertex while keeping
/// a small central aperture gap matching the InSightec ExAblate Neuro geometry.
const DEFAULT_CAP_MIN_POLAR_RAD: f64 = 0.22;
/// Default maximum polar angle from the superior vertex [rad].
///
/// 1.18 rad ≈ 67.6° — covers the calvarium without extending past the
/// temporoparietal junction, matching the InSightec ExAblate Neuro 4000.
const DEFAULT_CAP_MAX_POLAR_RAD: f64 = 1.18;
/// Unit metadata for geometry-only use of `BowlConfig`.
const BOWL_LAYOUT_UNIT_FREQUENCY_HZ: f64 = 1.0;
/// Unit metadata for geometry-only use of `BowlConfig`.
const BOWL_LAYOUT_UNIT_AMPLITUDE_PA: f64 = 1.0;
/// Clearance gap in metres between skull outer surface and the bowl shell.
const BOWL_SKIN_MARGIN_M: f64 = 0.015;
/// Minimum allowed bowl radius in metres (150 mm).
const BOWL_RADIUS_MIN_M: f64 = 0.150;
/// Number of beam paths sampled for visual and metric output.
const BEAM_SAMPLE_COUNT: usize = 72;
/// Number of DDA steps per beam for skull-intersection ray tracing.
const BEAM_TRACE_STEPS: usize = 320;
/// Fraction of z-range used at each end to estimate inferior/superior orientation.
const ORIENTATION_PROBE_FRACTION: f64 = 0.25;

#[derive(Clone, Debug)]
pub struct TranscranialFocusedBowlPlacement3D {
    pub head_surface_points_m: Vec<Point3>,
    pub skull_surface_points_m: Vec<Point3>,
    pub therapy_elements_m: Vec<Point3>,
    pub beam_start_points_m: Vec<Point3>,
    pub beam_end_points_m: Vec<Point3>,
    pub skull_intersections_m: Vec<Point3>,
    pub focus_m: Point3,
    pub bowl_radius_m: f64,
    pub intersection_fraction: f64,
}

/// Plan 3-D focused-bowl element placement from a head CT volume.
///
/// # Arguments
///
/// - `ct_hu` — 3-D array with shape `[NX, NY, NZ]`, values in Hounsfield units.
/// - `spacing_mm` — voxel spacing in millimetres for each axis.
/// - `element_count` — total number of therapy elements on the bowl shell.
/// - `surface_stride` — stride for thinning the surface point clouds (1 = dense).
/// - `body_hu_threshold` — HU threshold separating tissue from air (−350 HU).
/// - `skull_hu_threshold` — HU threshold separating bone from soft tissue (300 HU).
///
/// # Returns
///
/// Placement geometry centred at the calvarium centroid. All coordinates are in
/// metres and expressed relative to the calvarium centroid (origin = focus point).
pub fn plan_transcranial_focused_bowl_placement(
    ct_hu: &Array3<f64>,
    spacing_mm: [f64; 3],
    element_count: usize,
    surface_stride: usize,
    body_hu_threshold: f64,
    skull_hu_threshold: f64,
    target_fraction_xyz: Option<[f64; 3]>,
    scene_radius_m: Option<f64>,
    cap_min_polar_rad: Option<f64>,
    cap_max_polar_rad: Option<f64>,
) -> KwaversResult<TranscranialFocusedBowlPlacement3D> {
    if element_count < 16 {
        return Err(KwaversError::InvalidInput(
            "3-D focused-bowl placement requires at least 16 elements".to_owned(),
        ));
    }
    if spacing_mm
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(KwaversError::InvalidInput(
            "3-D focused-bowl placement requires positive finite CT spacing".to_owned(),
        ));
    }
    let body_mask = ct_hu.mapv(|hu| hu.is_finite() && hu >= body_hu_threshold);
    let skull_mask = ct_hu.mapv(|hu| hu.is_finite() && hu >= skull_hu_threshold);
    let brain_mask =
        ct_hu.mapv(|hu| hu.is_finite() && hu >= body_hu_threshold && hu < skull_hu_threshold);
    let target_support = if brain_mask.iter().any(|active| *active) {
        &brain_mask
    } else {
        &body_mask
    };
    let bounds = body_bounds(&body_mask)?;
    let areas = axial_areas(&body_mask);
    let nz = areas.len();
    let peak_z = areas
        .iter()
        .enumerate()
        .max_by_key(|(_, area)| **area)
        .map(|(idx, _)| idx)
        .unwrap_or((bounds.min[2] + bounds.max[2]) / 2);

    // Estimate which end of the z-axis is superior (vertex of skull = small area).
    // Compare summed area in the bottom ORIENTATION_PROBE_FRACTION of slices against
    // the top ORIENTATION_PROBE_FRACTION. The end with MORE area is inferior
    // (jaw/neck has larger cross-section than vertex).
    let probe = ((nz as f64 * ORIENTATION_PROBE_FRACTION) as usize).max(1);
    let inferior_sum: usize = areas[..probe.min(nz)].iter().sum();
    let superior_sum: usize = areas[nz.saturating_sub(probe)..].iter().sum();
    // superior_positive = true  means z = 0 is inferior, z = max is superior.
    // superior_positive = false means z = 0 is superior, z = max is inferior.
    let superior_positive = inferior_sum > superior_sum;

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
    if calvarium_max_z <= calvarium_min_z {
        return Err(KwaversError::InvalidInput(
            "Calvarium region is empty after orientation detection; verify CT covers the head."
                .to_owned(),
        ));
    }

    let center_index = if let Some(fraction) = target_fraction_xyz {
        let target_index = target_index_from_mask_fraction_3d(target_support, fraction)?;
        [
            target_index[0] as f64,
            target_index[1] as f64,
            target_index[2] as f64,
        ]
    } else {
        centroid_float(&body_mask, Some((calvarium_min_z, calvarium_max_z))).unwrap_or([
            0.5 * (bounds.min[0] + bounds.max[0]) as f64,
            0.5 * (bounds.min[1] + bounds.max[1]) as f64,
            0.5 * (calvarium_min_z + calvarium_max_z) as f64,
        ])
    };
    let spacing_m = [
        spacing_mm[0] * 1.0e-3,
        spacing_mm[1] * 1.0e-3,
        spacing_mm[2] * 1.0e-3,
    ];
    // The focal point is placed at the calvarium centroid (coordinate origin).
    // All output coordinates are expressed relative to this origin.
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
    // Bowl shell radius = maximum 3-D distance from the calvarium centroid to any
    // calvarium skull voxel, plus a clearance margin. This keeps the spherical
    // source surface outside the skull across the full calvarium region.
    let head_radius = body_radius(
        &body_mask,
        spacing_m,
        center_index,
        calvarium_min_z,
        calvarium_max_z,
    );
    let requested_radius_m = match scene_radius_m {
        Some(radius) if radius.is_finite() && radius > 0.0 => radius,
        Some(_) => {
            return Err(KwaversError::InvalidInput(
                "scene_radius_m must be positive and finite".to_owned(),
            ));
        }
        None => BOWL_RADIUS_MIN_M,
    };
    let bowl_radius_m = (head_radius + BOWL_SKIN_MARGIN_M)
        .max(requested_radius_m)
        .max(BOWL_RADIUS_MIN_M);
    let theta_min = cap_min_polar_rad
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(DEFAULT_CAP_MIN_POLAR_RAD);
    let theta_max = cap_max_polar_rad
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_CAP_MAX_POLAR_RAD);
    let therapy_elements_m = calvarium_cap_elements(
        element_count,
        bowl_radius_m,
        focus,
        superior_positive,
        theta_min,
        theta_max,
    )?;
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

    Ok(TranscranialFocusedBowlPlacement3D {
        head_surface_points_m,
        skull_surface_points_m,
        therapy_elements_m,
        beam_start_points_m,
        beam_end_points_m,
        skull_intersections_m,
        focus_m: focus,
        bowl_radius_m,
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
            "3-D focused-bowl placement found no CT head support voxels".to_owned(),
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
        .filter(|&((ix, iy, iz), active)| {
            *active
                && iz >= min_z
                && iz <= max_z
                && is_boundary_3d(mask, ix, iy, iz)
                && (ix + iy + iz) % stride == 0
        })
        .map(|((ix, iy, iz), _)| voxel_to_point(ix, iy, iz, spacing_m, center_index))
        .collect()
}

/// Compute the maximum 3-D distance from the calvarium centroid to any body voxel
/// in the calvarium region. This equals the spherical shell radius needed to fully
/// enclose the calvarium.
fn body_radius(
    mask: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
    min_z: usize,
    max_z: usize,
) -> f64 {
    mask.indexed_iter()
        .filter(|&((_, _, iz), active)| *active && iz >= min_z && iz <= max_z)
        .map(|((ix, iy, iz), _)| {
            let point = voxel_to_point(ix, iy, iz, spacing_m, center_index);
            (point.x_m * point.x_m + point.y_m * point.y_m + point.z_m * point.z_m).sqrt()
        })
        .fold(0.0_f64, f64::max)
}

/// Distribute `count` therapy elements on a focused-bowl calvarium cap.
///
/// # Arguments
///
/// - `theta_min_rad` — minimum polar angle from the superior vertex [rad].
///   Avoids element crowding at the vertex; typically 0.2–0.3 rad.
/// - `theta_max_rad` — maximum polar angle from the superior vertex [rad].
///   Determines how far the cap extends from the vertex; 1.18 rad ≈ 67.6°
///   covers the calvarium without reaching the temporoparietal junction.
///
/// The `superior_positive` flag flips the z-axis direction so the cap is oriented
/// toward the anatomical superior pole while the source-domain bowl layout owns
/// the equal-area sampling.
///
/// # Mathematical contract
///
/// The source-domain [`BowlTransducer`] owns polar-bound validation and
/// equal-area spherical-cap sampling. Clinical placement supplies only the
/// anatomy-derived vertex orientation and requested polar span.
fn calvarium_cap_elements(
    count: usize,
    radius_m: f64,
    focus: Point3,
    superior_positive: bool,
    theta_min_rad: f64,
    theta_max_rad: f64,
) -> KwaversResult<Vec<Point3>> {
    let sign = if superior_positive { 1.0_f64 } else { -1.0_f64 };
    let vertex_m = [focus.x_m, focus.y_m, focus.z_m + sign * radius_m];
    let focus_m = [focus.x_m, focus.y_m, focus.z_m];
    let config = BowlConfig::from_vertex_focus(
        vertex_m,
        focus_m,
        2.0 * radius_m,
        BOWL_LAYOUT_UNIT_FREQUENCY_HZ,
        BOWL_LAYOUT_UNIT_AMPLITUDE_PA,
    );
    let layout = BowlTransducer::with_polar_bounds(config, theta_min_rad, theta_max_rad, count)?;

    Ok(layout
        .element_positions()
        .iter()
        .map(|position| Point3 {
            x_m: position[0],
            y_m: position[1],
            z_m: position[2],
        })
        .collect())
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

fn sample_beams(
    elements: &[Point3],
    focus: Point3,
    skull_mask: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
) -> (Vec<Point3>, Vec<Point3>, Vec<Point3>) {
    let beam_count = BEAM_SAMPLE_COUNT.min(elements.len()).max(1);
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
    for step in 0..=BEAM_TRACE_STEPS {
        let t = step as f64 / BEAM_TRACE_STEPS as f64;
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
