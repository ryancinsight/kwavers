//! Finite-area source stencil geometry for tissue-boundary and exterior-coupling sources.

use std::f64::consts::PI;

use crate::clinical::therapy::theranostic_guidance::{
    abdominal3d::bowl::BOWL_THETA_MAX_RAD,
    nonlinear3d::types::{flat_index, GridIndex, Nonlinear3dAperture, SourceDomain},
};
use crate::core::constants::numerical::TWO_PI;

pub(super) fn finite_source_stencil(
    idx: GridIndex,
    n: usize,
    spacing_m: f64,
    aperture: &Nonlinear3dAperture,
    source_body_mask: Option<&[bool]>,
) -> Vec<(usize, f64)> {
    if aperture.source_domain == SourceDomain::ExteriorCoupling {
        return exterior_finite_area_stencil(idx, n, spacing_m, aperture, source_body_mask);
    }
    let offsets = [
        (0, 0, 0, 1.0),
        (-1, 0, 0, 0.5),
        (1, 0, 0, 0.5),
        (0, -1, 0, 0.5),
        (0, 1, 0, 0.5),
        (0, 0, -1, 0.5),
        (0, 0, 1, 0.5),
    ];
    let mut entries = Vec::with_capacity(offsets.len());
    for (dx, dy, dz, weight) in offsets {
        let x = idx.x as isize + dx;
        let y = idx.y as isize + dy;
        let z = idx.z as isize + dz;
        if x <= 0
            || y <= 0
            || z <= 0
            || x >= (n - 1) as isize
            || y >= (n - 1) as isize
            || z >= (n - 1) as isize
        {
            continue;
        }
        let cell = flat_index(
            GridIndex {
                x: x as usize,
                y: y as usize,
                z: z as usize,
            },
            n,
        );
        entries.push((cell, weight));
    }
    if entries.is_empty() {
        return vec![(flat_index(idx, n), 1.0)];
    }
    normalize_pressure_stencil(entries)
}

fn exterior_finite_area_stencil(
    idx: GridIndex,
    n: usize,
    spacing_m: f64,
    aperture: &Nonlinear3dAperture,
    source_body_mask: Option<&[bool]>,
) -> Vec<(usize, f64)> {
    let body = source_body_mask.expect("exterior coupling source stencil requires body mask");
    let radius_cells = exterior_element_radius_cells(aperture, spacing_m);
    let radius_m = radius_cells * spacing_m;
    let axial_sigma_m = (0.5 * spacing_m).max(1.0e-12);
    let tangent_sigma_m = (0.5 * radius_m).max(0.5 * spacing_m);
    let axis = unit_grid_direction(idx, aperture.focus);
    let search = radius_cells.ceil() as isize + 1;
    let mut entries = Vec::new();
    for dx in -search..=search {
        for dy in -search..=search {
            for dz in -search..=search {
                let x = idx.x as isize + dx;
                let y = idx.y as isize + dy;
                let z = idx.z as isize + dz;
                if x <= 0
                    || y <= 0
                    || z <= 0
                    || x >= (n - 1) as isize
                    || y >= (n - 1) as isize
                    || z >= (n - 1) as isize
                {
                    continue;
                }
                let cell = flat_index(
                    GridIndex {
                        x: x as usize,
                        y: y as usize,
                        z: z as usize,
                    },
                    n,
                );
                if body[cell] {
                    continue;
                }
                let offset_m = [
                    dx as f64 * spacing_m,
                    dy as f64 * spacing_m,
                    dz as f64 * spacing_m,
                ];
                let axial_m = dot(offset_m, axis);
                let radius2_m = dot(offset_m, offset_m);
                let tangent2_m = (radius2_m - axial_m * axial_m).max(0.0);
                if tangent2_m.sqrt() > radius_m || axial_m.abs() > spacing_m {
                    continue;
                }
                let weight = (-0.5 * tangent2_m / (tangent_sigma_m * tangent_sigma_m)).exp()
                    * (-0.5 * axial_m * axial_m / (axial_sigma_m * axial_sigma_m)).exp();
                entries.push((cell, weight));
            }
        }
    }
    if entries.is_empty() {
        let source_cell = flat_index(idx, n);
        assert!(
            !body[source_cell],
            "exterior coupling source stencil has no non-body support at {:?}",
            idx
        );
        return vec![(source_cell, 1.0)];
    }
    normalize_pressure_stencil(entries)
}

fn exterior_element_radius_cells(aperture: &Nonlinear3dAperture, spacing_m: f64) -> f64 {
    let count = aperture.sources.len().max(1) as f64;
    let mean_bowl_radius_m = aperture
        .sources
        .iter()
        .map(|source| grid_distance(*source, aperture.focus) * spacing_m)
        .sum::<f64>()
        / count;
    let cap_area_m2 =
        TWO_PI * mean_bowl_radius_m * mean_bowl_radius_m * (1.0 - BOWL_THETA_MAX_RAD.cos());
    let element_radius_m = (cap_area_m2 / (count * PI)).sqrt();
    (element_radius_m / spacing_m).clamp(1.0, 4.0)
}

// Pressure-boundary transducers impose pressure over an element face; the
// aperture apodization peak therefore stays at unit drive as grid support changes.
fn normalize_pressure_stencil(entries: Vec<(usize, f64)>) -> Vec<(usize, f64)> {
    let peak = entries
        .iter()
        .map(|(_, weight)| *weight)
        .fold(0.0, f64::max);
    if peak <= 0.0 || !peak.is_finite() {
        return entries;
    }
    entries
        .into_iter()
        .map(|(cell, weight)| (cell, weight / peak))
        .collect()
}

fn grid_distance(a: GridIndex, b: GridIndex) -> f64 {
    let dx = a.x as f64 - b.x as f64;
    let dy = a.y as f64 - b.y as f64;
    let dz = a.z as f64 - b.z as f64;
    dx.hypot(dy).hypot(dz)
}

fn unit_grid_direction(source: GridIndex, focus: GridIndex) -> [f64; 3] {
    let raw = [
        focus.x as f64 - source.x as f64,
        focus.y as f64 - source.y as f64,
        focus.z as f64 - source.z as f64,
    ];
    let norm = raw[0].hypot(raw[1]).hypot(raw[2]).max(1.0e-12);
    [raw[0] / norm, raw[1] / norm, raw[2] / norm]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}
