//! Shared helpers for the cavitation forward and inverse modules.

use leto::Array3;

use super::super::types::{flat_index, GridIndex};

pub(super) fn active_indices(body: &[bool]) -> Vec<usize> {
    body.iter()
        .enumerate()
        .filter_map(|(idx, active)| active.then_some(idx))
        .collect()
}

pub(super) fn normalize(values: &[f64], body: &[bool]) -> Vec<f64> {
    let peak = values
        .iter()
        .zip(body.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .fold(0.0, f64::max)
        .max(1.0e-12);
    values
        .iter()
        .zip(body.iter())
        .map(|(value, active)| active.then_some(*value / peak).unwrap_or(0.0))
        .collect()
}

pub(super) fn grid_index(flat: usize, n: usize) -> GridIndex {
    GridIndex {
        x: flat / (n * n),
        y: (flat / n) % n,
        z: flat % n,
    }
}

pub(super) fn unflatten(values: &[f64], n: usize) -> Array3<f64> {
    Array3::from_shape_fn((n, n, n), |[x, y, z]| {
        values[flat_index(GridIndex { x, y, z }, n)]
    })
}
