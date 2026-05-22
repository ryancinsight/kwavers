//! Focused abdominal bowl placement for the nonlinear 3-D solver.

use std::collections::HashSet;

use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

use super::super::abdominal3d::bowl::{bowl_elements, BOWL_THETA_MAX_RAD};
use super::super::abdominal3d::helpers::{exterior_air_mask, nearest_exterior_skin_point};
use super::super::Point3;
use super::types::{grid_point_m, GridIndex, Nonlinear3dVolume};

const MIN_BOWL_RADIUS_M: f64 = 0.060;
const BOWL_OVERSAMPLE: usize = 8;

pub(crate) fn abdominal_bowl_radius_m(focal_depth_m: f64) -> f64 {
    (focal_depth_m / BOWL_THETA_MAX_RAD.cos()).max(MIN_BOWL_RADIUS_M)
}

#[derive(Clone, Copy)]
struct ExteriorCell {
    index: GridIndex,
    point_m: Point3,
}

pub(super) fn abdominal_bowl_candidates(
    volume: &Nonlinear3dVolume,
    requested_count: usize,
) -> KwaversResult<Vec<GridIndex>> {
    let n = volume.body_mask.dim().0;
    let exterior = exterior_air_mask(&volume.body_mask);
    let focus_m = grid_point_m(volume.focus, n, volume.spacing_m);
    let center = [0.5 * (n - 1) as f64; 3];
    let skin_m = if let Some(skin) = volume.aperture_skin {
        grid_point_m(skin, n, volume.spacing_m)
    } else {
        nearest_exterior_skin_point(
            &volume.body_mask,
            &exterior,
            [volume.spacing_m; 3],
            center,
            focus_m,
        )?
    };
    let focal_depth_m = distance(focus_m, skin_m);
    if focal_depth_m <= volume.spacing_m || !focal_depth_m.is_finite() {
        return Err(KwaversError::InvalidInput(
            "nonlinear abdominal aperture has degenerate skin-to-target depth".to_owned(),
        ));
    }
    let radius_m = abdominal_bowl_radius_m(focal_depth_m);
    let skin_outward = normalize([
        skin_m.x_m - focus_m.x_m,
        skin_m.y_m - focus_m.y_m,
        skin_m.z_m - focus_m.z_m,
    ]);
    let planned_outward = volume
        .aperture_direction
        .map(normalize)
        .unwrap_or(skin_outward);
    let outward = if dot(planned_outward, skin_outward) > 0.5 {
        planned_outward
    } else {
        skin_outward
    };
    let cells = source_eligible_exterior_cells(volume, &exterior, focus_m, outward);
    if cells.len() < 4 {
        return Err(KwaversError::InvalidInput(
            "nonlinear abdominal aperture has fewer than four exterior coupling cells".to_owned(),
        ));
    }

    let ideal_count = requested_count.max(1) * BOWL_OVERSAMPLE;
    let ideal_points = bowl_elements(ideal_count, skin_m, focus_m, radius_m)?;
    let mut seen = HashSet::with_capacity(requested_count);
    let mut selected = Vec::with_capacity(requested_count);
    for point in ideal_points {
        if let Some(cell) = nearest_cell(&cells, point) {
            if seen.insert((cell.index.x, cell.index.y, cell.index.z)) {
                selected.push(cell.index);
                if selected.len() == requested_count {
                    return Ok(selected);
                }
            }
        }
    }

    let mut ranked = cells;
    ranked.sort_by(|a, b| {
        bowl_rank(*a, focus_m, outward, radius_m)
            .total_cmp(&bowl_rank(*b, focus_m, outward, radius_m))
    });
    for cell in ranked {
        if seen.insert((cell.index.x, cell.index.y, cell.index.z)) {
            selected.push(cell.index);
            if selected.len() == requested_count {
                break;
            }
        }
    }
    Ok(selected)
}

fn source_eligible_exterior_cells(
    volume: &Nonlinear3dVolume,
    exterior: &Array3<bool>,
    focus_m: Point3,
    outward: [f64; 3],
) -> Vec<ExteriorCell> {
    let n = volume.body_mask.dim().0;
    let cos_limit = (BOWL_THETA_MAX_RAD.cos() - 0.15).max(0.0);
    let mut cells = Vec::new();
    for ((x, y, z), is_exterior) in exterior.indexed_iter() {
        if !*is_exterior
            || volume.body_mask[[x, y, z]]
            || x == 0
            || y == 0
            || z == 0
            || x + 1 >= n
            || y + 1 >= n
            || z + 1 >= n
        {
            continue;
        }
        let index = GridIndex { x, y, z };
        let point_m = grid_point_m(index, n, volume.spacing_m);
        let direction = normalize([
            point_m.x_m - focus_m.x_m,
            point_m.y_m - focus_m.y_m,
            point_m.z_m - focus_m.z_m,
        ]);
        if dot(direction, outward) >= cos_limit {
            cells.push(ExteriorCell { index, point_m });
        }
    }
    cells
}

fn nearest_cell(cells: &[ExteriorCell], point: Point3) -> Option<ExteriorCell> {
    cells
        .iter()
        .copied()
        .min_by(|a, b| distance2(a.point_m, point).total_cmp(&distance2(b.point_m, point)))
}

fn bowl_rank(cell: ExteriorCell, focus_m: Point3, outward: [f64; 3], radius_m: f64) -> f64 {
    let v = [
        cell.point_m.x_m - focus_m.x_m,
        cell.point_m.y_m - focus_m.y_m,
        cell.point_m.z_m - focus_m.z_m,
    ];
    let distance_m = v[0].hypot(v[1]).hypot(v[2]).max(1.0e-12);
    let direction = [v[0] / distance_m, v[1] / distance_m, v[2] / distance_m];
    let cos_theta = dot(direction, outward).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let cap_mid = 0.5 * BOWL_THETA_MAX_RAD;
    (theta - cap_mid).abs() + 0.25 * ((distance_m - radius_m).abs() / radius_m.max(1.0e-12))
}

fn distance(a: Point3, b: Point3) -> f64 {
    distance2(a, b).sqrt()
}

fn distance2(a: Point3, b: Point3) -> f64 {
    let dx = a.x_m - b.x_m;
    let dy = a.y_m - b.y_m;
    let dz = a.z_m - b.z_m;
    dx.mul_add(dx, dy.mul_add(dy, dz * dz))
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let norm = v[0].hypot(v[1]).hypot(v[2]).max(1.0e-12);
    [v[0] / norm, v[1] / norm, v[2] / norm]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use crate::clinical::therapy::theranostic_guidance::AnatomyKind;
    use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use super::super::types::Nonlinear3dVolume;
    use super::*;

    #[test]
    fn abdominal_bowl_candidates_are_exterior_and_target_facing() {
        let n = 24;
        let focus = GridIndex {
            x: 12,
            y: 14,
            z: 12,
        };
        let mut body = Array3::from_elem((n, n, n), false);
        for x in 4..20 {
            for y in 4..20 {
                for z in 4..20 {
                    body[[x, y, z]] = true;
                }
            }
        }
        let mut target = Array3::from_elem((n, n, n), false);
        target[[focus.x, focus.y, focus.z]] = true;
        let volume = test_volume(body, target, focus);

        let candidates = abdominal_bowl_candidates(&volume, 32).unwrap();

        assert_eq!(candidates.len(), 32);
        for source in candidates {
            assert!(
                !volume.body_mask[[source.x, source.y, source.z]],
                "source must be in exterior coupling medium: {source:?}"
            );
            assert!(
                source.x > 0
                    && source.y > 0
                    && source.z > 0
                    && source.x + 1 < n
                    && source.y + 1 < n
                    && source.z + 1 < n
            );
            assert!(
                source.y > focus.y,
                "source must stay on the planned skin side of the target: {source:?}"
            );
        }
    }

    fn test_volume(
        body_mask: Array3<bool>,
        target_mask: Array3<bool>,
        focus: GridIndex,
    ) -> Nonlinear3dVolume {
        let n = body_mask.dim().0;
        let f64_zeros = Array3::from_elem((n, n, n), 0.0);
        let speed = Array3::from_elem((n, n, n), SOUND_SPEED_WATER_SIM);
        Nonlinear3dVolume {
            anatomy: AnatomyKind::Liver,
            ct_hu: f64_zeros.clone(),
            label: Array3::from_elem((n, n, n), 0),
            body_mask,
            target_mask,
            inversion_mask: Array3::from_elem((n, n, n), true),
            density_kg_m3: Array3::from_elem((n, n, n), DENSITY_WATER_NOMINAL),
            background_beta: Array3::from_elem((n, n, n), 3.5),
            true_beta: Array3::from_elem((n, n, n), 3.5),
            background_sound_speed_m_s: speed.clone(),
            true_sound_speed_m_s: speed,
            attenuation_np_per_m_mhz: f64_zeros.clone(),
            attenuation_power_law_y: Array3::from_elem((n, n, n), 1.0),
            spacing_m: 1.0e-3,
            source_dimensions: [n, n, n],
            source_spacing_m: [1.0e-3; 3],
            crop_bounds_index: [0, n - 1, 0, n - 1, 0, n - 1],
            aperture_direction: Some([0.0, 1.0, 0.0]),
            aperture_skin: Some(GridIndex {
                x: 12,
                y: 20,
                z: 12,
            }),
            focus,
        }
    }
}
