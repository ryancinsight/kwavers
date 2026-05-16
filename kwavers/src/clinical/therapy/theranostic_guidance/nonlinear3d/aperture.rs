//! 3-D same-aperture source/receiver placement on CT-derived body support.

use std::cmp::Ordering;

use crate::core::error::{KwaversError, KwaversResult};

use super::super::abdominal3d::helpers::{exterior_air_mask, nearest_exterior_skin_point};
use super::super::AnatomyKind;
use super::super::Point3;
use super::types::{
    grid_point_m, GridIndex, Nonlinear3dAperture, Nonlinear3dConfig, Nonlinear3dVolume,
};
use super::volume::centroid_index;

pub(crate) fn build_aperture(
    volume: &Nonlinear3dVolume,
    config: &Nonlinear3dConfig,
) -> KwaversResult<Nonlinear3dAperture> {
    let candidates = match volume.anatomy {
        AnatomyKind::Brain => brain_candidates(volume, config.element_count),
        AnatomyKind::Liver | AnatomyKind::Kidney => abdominal_candidates(volume),
    };
    if candidates.len() < 4 {
        return Err(KwaversError::InvalidInput(
            "nonlinear 3-D aperture found fewer than four skin boundary cells".to_owned(),
        ));
    }
    let sources = select_evenly(candidates, config.element_count);
    let receivers = select_evenly(sources.clone(), config.receiver_count);
    let n = volume.body_mask.dim().0;
    let therapy_points_m = sources
        .iter()
        .map(|idx| grid_point_m(*idx, n, volume.spacing_m))
        .collect();
    let receiver_points_m = receivers
        .iter()
        .map(|idx| grid_point_m(*idx, n, volume.spacing_m))
        .collect();
    let model_name = match volume.anatomy {
        AnatomyKind::Brain => "insightec_like_calvarium_helmet_3d_westervelt_sources",
        AnatomyKind::Liver => "liver_histosonics_like_skin_coupled_3d_westervelt_sources",
        AnatomyKind::Kidney => "kidney_histosonics_like_skin_coupled_3d_westervelt_sources",
    }
    .to_owned();
    Ok(Nonlinear3dAperture {
        sources,
        receivers,
        therapy_points_m,
        receiver_points_m,
        model_name,
        focus: volume.focus,
    })
}

fn brain_candidates(volume: &Nonlinear3dVolume, requested_count: usize) -> Vec<GridIndex> {
    let n = volume.body_mask.dim().0;
    let peak_z = axial_peak(&volume.body_mask);
    let target = centroid_index(&volume.target_mask).unwrap_or(volume.focus);
    let superior_positive = peak_z <= n / 2;
    let exterior = exterior_air_mask(&volume.body_mask);
    let boundary = exterior_boundary_cells(&volume.body_mask, &exterior);
    let mut candidates = boundary
        .iter()
        .copied()
        .filter(|idx| is_brain_cap_cell(*idx, peak_z, target, superior_positive))
        .filter(|idx| volume.ct_hu[[idx.x, idx.y, idx.z]] >= 250.0)
        .collect::<Vec<_>>();
    if candidates.len() < requested_count {
        candidates.extend(
            boundary
                .into_iter()
                .filter(|idx| is_brain_cap_cell(*idx, peak_z, target, superior_positive)),
        );
        candidates.sort_by_key(|idx| (idx.x, idx.y, idx.z));
        candidates.dedup();
    }
    if candidates.len() < 8 {
        candidates = boundary_cells(&volume.body_mask);
    }
    sort_by_spherical_angle(&mut candidates, target);
    candidates
}

fn is_brain_cap_cell(
    idx: GridIndex,
    peak_z: usize,
    target: GridIndex,
    superior_positive: bool,
) -> bool {
    if superior_positive {
        idx.z >= peak_z && idx.z >= target.z
    } else {
        idx.z <= peak_z && idx.z <= target.z
    }
}

fn abdominal_candidates(volume: &Nonlinear3dVolume) -> Vec<GridIndex> {
    let focus = volume.focus;
    let exterior = exterior_air_mask(&volume.body_mask);
    let skin = exterior_skin_grid_index(volume, &exterior)
        .or_else(|| nearest_boundary(&volume.body_mask, focus))
        .unwrap_or(focus);
    let direction = volume
        .aperture_direction
        .unwrap_or_else(|| unit_vector(focus, skin));
    let mut candidates = exterior_boundary_cells(&volume.body_mask, &exterior)
        .into_iter()
        .filter(|idx| dot(unit_vector(focus, *idx), direction) > 0.25)
        .collect::<Vec<_>>();
    if candidates.len() < 8 {
        candidates = boundary_cells(&volume.body_mask);
    }
    candidates.sort_by(|a, b| {
        angle_about_direction(*a, focus, direction)
            .total_cmp(&angle_about_direction(*b, focus, direction))
    });
    candidates
}

fn exterior_skin_grid_index(
    volume: &Nonlinear3dVolume,
    exterior: &ndarray::Array3<bool>,
) -> Option<GridIndex> {
    let n = volume.body_mask.dim().0;
    let center = [0.5 * (n - 1) as f64; 3];
    let spacing = [volume.spacing_m; 3];
    let focus_m = grid_point_m(volume.focus, n, volume.spacing_m);
    nearest_exterior_skin_point(&volume.body_mask, exterior, spacing, center, focus_m)
        .ok()
        .map(|point| point_to_grid_index(point, n, volume.spacing_m))
}

fn point_to_grid_index(point: Point3, n: usize, spacing_m: f64) -> GridIndex {
    let center = 0.5 * (n - 1) as f64;
    let to_index = |value_m: f64| -> usize {
        (value_m / spacing_m + center)
            .round()
            .clamp(0.0, (n - 1) as f64) as usize
    };
    GridIndex {
        x: to_index(point.x_m),
        y: to_index(point.y_m),
        z: to_index(point.z_m),
    }
}

fn exterior_boundary_cells(
    mask: &ndarray::Array3<bool>,
    exterior: &ndarray::Array3<bool>,
) -> Vec<GridIndex> {
    let mut out = Vec::new();
    for ((x, y, z), active) in mask.indexed_iter() {
        if *active && touches_exterior(mask, exterior, x, y, z) {
            out.push(GridIndex { x, y, z });
        }
    }
    out
}

fn touches_exterior(
    mask: &ndarray::Array3<bool>,
    exterior: &ndarray::Array3<bool>,
    x: usize,
    y: usize,
    z: usize,
) -> bool {
    let (nx, ny, nz) = mask.dim();
    (x > 0 && !mask[[x - 1, y, z]] && exterior[[x - 1, y, z]])
        || (x + 1 < nx && !mask[[x + 1, y, z]] && exterior[[x + 1, y, z]])
        || (y > 0 && !mask[[x, y - 1, z]] && exterior[[x, y - 1, z]])
        || (y + 1 < ny && !mask[[x, y + 1, z]] && exterior[[x, y + 1, z]])
        || (z > 0 && !mask[[x, y, z - 1]] && exterior[[x, y, z - 1]])
        || (z + 1 < nz && !mask[[x, y, z + 1]] && exterior[[x, y, z + 1]])
}

fn boundary_cells(mask: &ndarray::Array3<bool>) -> Vec<GridIndex> {
    let mut out = Vec::new();
    for ((x, y, z), active) in mask.indexed_iter() {
        if *active && is_boundary(mask, x, y, z) {
            out.push(GridIndex { x, y, z });
        }
    }
    out
}

fn is_boundary(mask: &ndarray::Array3<bool>, x: usize, y: usize, z: usize) -> bool {
    let (nx, ny, nz) = mask.dim();
    x == 0
        || y == 0
        || z == 0
        || x + 1 == nx
        || y + 1 == ny
        || z + 1 == nz
        || !mask[[x - 1, y, z]]
        || !mask[[x + 1, y, z]]
        || !mask[[x, y - 1, z]]
        || !mask[[x, y + 1, z]]
        || !mask[[x, y, z - 1]]
        || !mask[[x, y, z + 1]]
}

fn axial_peak(mask: &ndarray::Array3<bool>) -> usize {
    let (_, _, nz) = mask.dim();
    (0..nz)
        .max_by_key(|z| {
            let mut count = 0usize;
            for x in 0..mask.dim().0 {
                for y in 0..mask.dim().1 {
                    count += usize::from(mask[[x, y, *z]]);
                }
            }
            count
        })
        .unwrap_or(nz / 2)
}

fn nearest_boundary(mask: &ndarray::Array3<bool>, focus: GridIndex) -> Option<GridIndex> {
    boundary_cells(mask).into_iter().min_by(|a, b| {
        distance2(*a, focus)
            .partial_cmp(&distance2(*b, focus))
            .unwrap_or(Ordering::Equal)
    })
}

fn select_evenly(mut candidates: Vec<GridIndex>, count: usize) -> Vec<GridIndex> {
    candidates.sort_by_key(|idx| (idx.x, idx.y, idx.z));
    candidates.dedup();
    if candidates.len() <= count {
        return candidates;
    }
    let mut selected = Vec::with_capacity(count);
    for k in 0..count {
        let idx = k * candidates.len() / count;
        selected.push(candidates[idx]);
    }
    selected.sort_by_key(|idx| (idx.x, idx.y, idx.z));
    selected.dedup();
    selected
}

fn sort_by_spherical_angle(candidates: &mut [GridIndex], focus: GridIndex) {
    candidates.sort_by(|a, b| {
        let aa = (a.y as f64 - focus.y as f64).atan2(a.x as f64 - focus.x as f64);
        let bb = (b.y as f64 - focus.y as f64).atan2(b.x as f64 - focus.x as f64);
        aa.total_cmp(&bb)
    });
}

fn distance2(a: GridIndex, b: GridIndex) -> f64 {
    let dx = a.x as f64 - b.x as f64;
    let dy = a.y as f64 - b.y as f64;
    let dz = a.z as f64 - b.z as f64;
    dx * dx + dy * dy + dz * dz
}

fn unit_vector(a: GridIndex, b: GridIndex) -> [f64; 3] {
    let raw = [
        b.x as f64 - a.x as f64,
        b.y as f64 - a.y as f64,
        b.z as f64 - a.z as f64,
    ];
    let norm = raw[0].hypot(raw[1]).hypot(raw[2]).max(1.0e-12);
    [raw[0] / norm, raw[1] / norm, raw[2] / norm]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn angle_about_direction(idx: GridIndex, focus: GridIndex, direction: [f64; 3]) -> f64 {
    let v = unit_vector(focus, idx);
    let axis = if direction[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u = normalize(cross(direction, axis));
    let w = cross(direction, u);
    dot(v, w).atan2(dot(v, u))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let norm = v[0].hypot(v[1]).hypot(v[2]).max(1.0e-12);
    [v[0] / norm, v[1] / norm, v[2] / norm]
}
