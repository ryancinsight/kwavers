//! 3-D same-aperture source/receiver placement on CT-derived body support.

use std::f64::consts::PI;

use crate::core::error::{KwaversError, KwaversResult};

use super::super::abdominal3d::helpers::exterior_air_mask;
use super::super::AnatomyKind;
use super::super::Point3;
use super::aperture_bowl::abdominal_bowl_candidates;
use super::types::{
    GridIndex, Nonlinear3dAperture, Nonlinear3dConfig, Nonlinear3dVolume, SourceDomain,
};
use super::volume::centroid_index;

pub(crate) fn build_aperture(
    volume: &Nonlinear3dVolume,
    config: &Nonlinear3dConfig,
) -> KwaversResult<Nonlinear3dAperture> {
    let candidates = match volume.anatomy {
        AnatomyKind::Brain => brain_candidates(volume, config.element_count),
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            abdominal_bowl_candidates(volume, config.element_count)?
        }
    };
    if candidates.len() < 4 {
        return Err(KwaversError::InvalidInput(
            "nonlinear 3-D aperture found fewer than four skin boundary cells".to_owned(),
        ));
    }
    let sources = match volume.anatomy {
        AnatomyKind::Brain => select_evenly(candidates, config.element_count),
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            select_evenly_preserving_order(candidates, config.element_count)
        }
    };
    let receivers = match volume.anatomy {
        AnatomyKind::Brain => select_evenly(sources.clone(), config.receiver_count),
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            select_evenly_preserving_order(sources.clone(), config.receiver_count)
        }
    };
    let n = volume.body_mask.dim().0;
    let therapy_points_m = sources
        .iter()
        .map(|idx| source_frame_grid_point_m(*idx, volume, n))
        .collect();
    let receiver_points_m = receivers
        .iter()
        .map(|idx| source_frame_grid_point_m(*idx, volume, n))
        .collect();
    let model_name = match volume.anatomy {
        AnatomyKind::Brain => "insightec_like_calvarium_focused_bowl_3d_westervelt_sources",
        AnatomyKind::Liver => {
            "liver_histosonics_like_focused_bowl_slowness_steered_3d_westervelt_sources"
        }
        AnatomyKind::Kidney => {
            "kidney_histosonics_like_focused_bowl_slowness_steered_3d_westervelt_sources"
        }
    }
    .to_owned();
    Ok(Nonlinear3dAperture {
        sources,
        receivers,
        therapy_points_m,
        receiver_points_m,
        model_name,
        source_domain: match volume.anatomy {
            AnatomyKind::Brain => SourceDomain::TissueBoundary,
            AnatomyKind::Liver | AnatomyKind::Kidney => SourceDomain::ExteriorCoupling,
        },
        focus: volume.focus,
    })
}

fn source_frame_grid_point_m(idx: GridIndex, volume: &Nonlinear3dVolume, n: usize) -> Point3 {
    let bounds = volume.crop_bounds_index;
    let x = source_axis_index(idx.x, bounds[0], bounds[1], n);
    let y = source_axis_index(idx.y, bounds[2], bounds[3], n);
    let z = source_axis_index(idx.z, bounds[4], bounds[5], n);
    Point3 {
        x_m: (x - 0.5 * (volume.source_dimensions[0] - 1) as f64) * volume.source_spacing_m[0],
        y_m: (y - 0.5 * (volume.source_dimensions[1] - 1) as f64) * volume.source_spacing_m[1],
        z_m: (z - 0.5 * (volume.source_dimensions[2] - 1) as f64) * volume.source_spacing_m[2],
    }
}

fn source_axis_index(index: usize, min: usize, max: usize, n: usize) -> f64 {
    if n <= 1 || max == min {
        return min as f64;
    }
    min as f64 + index as f64 * (max - min) as f64 / (n - 1) as f64
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
        .filter(|idx| volume.ct_hu[[idx.x, idx.y, idx.z]] >= 200.0) // include cancellous bone
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
    // Fibonacci sphere selection gives uniform angular coverage over the cap.
    fibonacci_sphere_select(candidates, requested_count, target)
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

/// Select at most `count` elements uniformly distributed over the hemispherical
/// cap using a Fibonacci (golden-angle) sphere lattice.
///
/// ## Theorem
/// The golden angle φ_g = 2π(1 − φ⁻²) ≈ 2.3999 rad (φ = golden ratio) minimises
/// the maximum nearest-neighbour angular gap for N points on a sphere (Álvarez 2001).
/// This guarantees that for a 1024-element transcranial array the element distribution
/// maximises the uniform illumination solid angle over the calvarium cap, reducing
/// coherent grating-lobe artefacts in RTM/FWI reconstructions.
///
/// For each of the N desired Fibonacci lattice positions, the candidate grid cell
/// whose `(θ, φ)` polar angles (relative to the focus) are closest is selected
/// without replacement.
///
/// # Reference
/// Álvarez (2001), arXiv:math/0106096.
fn fibonacci_sphere_select(
    candidates: Vec<GridIndex>,
    count: usize,
    focus: GridIndex,
) -> Vec<GridIndex> {
    if candidates.len() <= count {
        return candidates;
    }

    // Convert each candidate to polar angles (θ, φ) relative to the focus.
    let to_polar = |idx: &GridIndex| -> (f64, f64) {
        let dx = idx.x as f64 - focus.x as f64;
        let dy = idx.y as f64 - focus.y as f64;
        let dz = idx.z as f64 - focus.z as f64;
        let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
        let theta = (dz / r).clamp(-1.0, 1.0).acos(); // polar [0, π]
        let phi = dy.atan2(dx); // azimuth [−π, π]
        (theta, phi)
    };
    let polars: Vec<(f64, f64)> = candidates.iter().map(to_polar).collect();

    // Determine polar range spanned by the candidate cap.
    let theta_min = polars.iter().map(|(t, _)| *t).fold(f64::INFINITY, f64::min);
    let theta_max = polars.iter().map(|(t, _)| *t).fold(0.0_f64, f64::max);
    let phi_g = PI * (3.0 - 5.0_f64.sqrt()); // golden angle ≈ 2.3999 rad

    let mut selected = vec![false; candidates.len()];
    let mut result = Vec::with_capacity(count);

    for k in 0..count {
        // Fibonacci lattice point mapped to [theta_min, theta_max].
        let frac = k as f64 / (count as f64 - 1.0).max(1.0);
        let theta_ideal = theta_min + frac * (theta_max - theta_min);
        let phi_ideal = k as f64 * phi_g;

        // Nearest unselected candidate in angular (θ, φ) space.
        let best = polars
            .iter()
            .enumerate()
            .filter(|(i, _)| !selected[*i])
            .min_by(|(_, (ta, pa)), (_, (tb, pb))| {
                let da = (ta - theta_ideal).powi(2) + azimuth_diff_sq(*pa, phi_ideal);
                let db = (tb - theta_ideal).powi(2) + azimuth_diff_sq(*pb, phi_ideal);
                da.total_cmp(&db)
            })
            .map(|(i, _)| i);

        if let Some(best_idx) = best {
            selected[best_idx] = true;
            result.push(candidates[best_idx]);
        }
    }
    result.sort_by_key(|idx| (idx.x, idx.y, idx.z));
    result
}

/// Squared angular difference on the circle [−π, π), wrapping correctly.
#[inline]
fn azimuth_diff_sq(a: f64, b: f64) -> f64 {
    let diff = ((a - b + 3.0 * PI).rem_euclid(2.0 * PI)) - PI;
    diff * diff
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

fn select_evenly_preserving_order(mut candidates: Vec<GridIndex>, count: usize) -> Vec<GridIndex> {
    let mut seen = std::collections::HashSet::with_capacity(candidates.len());
    candidates.retain(|idx| seen.insert((idx.x, idx.y, idx.z)));
    if candidates.len() <= count {
        return candidates;
    }
    (0..count)
        .map(|k| candidates[k * candidates.len() / count])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{select_evenly_preserving_order, GridIndex};

    #[test]
    fn abdominal_selection_preserves_target_facing_angular_order() {
        let candidates = vec![
            GridIndex { x: 9, y: 0, z: 0 },
            GridIndex { x: 0, y: 9, z: 0 },
            GridIndex { x: 1, y: 0, z: 0 },
            GridIndex { x: 0, y: 1, z: 0 },
        ];

        let selected = select_evenly_preserving_order(candidates, 2);

        assert_eq!(
            selected,
            vec![
                GridIndex { x: 9, y: 0, z: 0 },
                GridIndex { x: 1, y: 0, z: 0 },
            ]
        );
    }

    #[test]
    fn abdominal_selection_removes_duplicates_without_reordering() {
        let candidates = vec![
            GridIndex { x: 2, y: 0, z: 0 },
            GridIndex { x: 1, y: 0, z: 0 },
            GridIndex { x: 2, y: 0, z: 0 },
        ];

        let selected = select_evenly_preserving_order(candidates, 4);

        assert_eq!(
            selected,
            vec![
                GridIndex { x: 2, y: 0, z: 0 },
                GridIndex { x: 1, y: 0, z: 0 },
            ]
        );
    }
}
