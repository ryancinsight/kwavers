//! Row / column dot-product kernels and shared lookup helpers.
//!
//! The four matvec / t_matvec kernels (`pitch_catch_row_dot`,
//! `passive_row_dot`, `pitch_catch_column_dot`, `passive_column_dot`) and the
//! two normal-equation diagonal kernels (`pitch_catch_column_norm_sq`,
//! `passive_column_norm_sq`) all share the same `column_lookup` and per-cell
//! attenuation/coordinate access pattern; co-locating them here keeps the
//! inner-loop arithmetic in one place for future SIMD / cache-aware tuning.

use super::super::active_grid::{ActiveGrid, PlanarPoint};
use super::super::finite_frequency::SameApertureMedium;
use super::types::{PassiveRow, PitchCatchRow};

#[inline]
pub(super) fn pitch_catch_row_dot(
    spec: &PitchCatchRow,
    inv_norm: f32,
    x: &[f32],
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) -> f32 {
    let mut sum = 0.0_f32;
    for (col, x_value) in x.iter().enumerate() {
        let (point, alpha) = column_lookup(active, medium, col);
        sum += spec.unscaled_value(point, alpha, medium.spacing_m) * inv_norm * *x_value;
    }
    sum
}

#[inline]
pub(super) fn passive_row_dot(
    spec: &PassiveRow,
    inv_norm: f32,
    x: &[f32],
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) -> f32 {
    let mut sum = 0.0_f32;
    for (col, x_value) in x.iter().enumerate() {
        let (point, alpha) = column_lookup(active, medium, col);
        sum += spec.unscaled_value(point, alpha, medium.spacing_m) * inv_norm * *x_value;
    }
    sum
}

#[inline]
pub(super) fn pitch_catch_column_dot(
    specs: &[PitchCatchRow],
    scaled_y: &[f32],
    col: usize,
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) -> f32 {
    let (point, alpha) = column_lookup(active, medium, col);
    let mut sum = 0.0_f32;
    for (spec, scaled) in specs.iter().zip(scaled_y.iter()) {
        sum += spec.unscaled_value(point, alpha, medium.spacing_m) * *scaled;
    }
    sum
}

#[inline]
pub(super) fn passive_column_dot(
    specs: &[PassiveRow],
    scaled_y: &[f32],
    col: usize,
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) -> f32 {
    let (point, alpha) = column_lookup(active, medium, col);
    let mut sum = 0.0_f32;
    for (spec, scaled) in specs.iter().zip(scaled_y.iter()) {
        sum += spec.unscaled_value(point, alpha, medium.spacing_m) * *scaled;
    }
    sum
}

#[inline]
pub(super) fn pitch_catch_column_norm_sq(
    specs: &[PitchCatchRow],
    inv_row_norms: &[f32],
    col: usize,
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) -> f32 {
    let (point, alpha) = column_lookup(active, medium, col);
    let mut sum = 0.0_f32;
    for (spec, inv_norm) in specs.iter().zip(inv_row_norms.iter()) {
        let value = spec.unscaled_value(point, alpha, medium.spacing_m) * *inv_norm;
        sum += value * value;
    }
    sum
}

#[inline]
pub(super) fn passive_column_norm_sq(
    specs: &[PassiveRow],
    inv_row_norms: &[f32],
    col: usize,
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) -> f32 {
    let (point, alpha) = column_lookup(active, medium, col);
    let mut sum = 0.0_f32;
    for (spec, inv_norm) in specs.iter().zip(inv_row_norms.iter()) {
        let value = spec.unscaled_value(point, alpha, medium.spacing_m) * *inv_norm;
        sum += value * value;
    }
    sum
}

#[inline]
pub(super) fn column_lookup(
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
    col: usize,
) -> (PlanarPoint, f64) {
    let point = active.points_m[col];
    let (ix, iy) = active.indices[col];
    let alpha = medium.attenuation_np_per_m_mhz[[ix, iy]];
    (point, alpha)
}

#[inline]
pub(super) fn scaled_input(y: &[f32], inv_row_norms: &[f32]) -> Vec<f32> {
    y.iter()
        .zip(inv_row_norms.iter())
        .map(|(yv, inv)| yv * inv)
        .collect()
}

#[inline]
pub(super) fn distance(a: PlanarPoint, b: PlanarPoint) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}
