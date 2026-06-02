//! Row-spec construction, per-row norms, and row writers used by
//! `materialize` and `row_values`.

use super::super::active_grid::{ActiveGrid, PlanarPoint};
use super::super::finite_frequency::{SameApertureMedium, SameApertureSettings, C_REF_M_S};
use super::dot::column_lookup;
use super::types::{OperatorKind, PassiveRow, PitchCatchRow};
use rayon::prelude::*;

pub(super) fn pitch_catch_rows(
    therapy_elements: &[PlanarPoint],
    settings: SameApertureSettings<'_>,
    harmonic: f64,
) -> Vec<PitchCatchRow> {
    let nonlinear_path_weight = harmonic > 1.0;
    let mut rows = Vec::with_capacity(
        settings.frequencies_hz.len() * settings.receiver_offsets.len() * therapy_elements.len(),
    );
    for &frequency_hz in settings.frequencies_hz {
        let k = std::f64::consts::TAU * frequency_hz * harmonic / settings.phase_speed_m_s;
        let frequency_mhz = frequency_hz * 1.0e-6 * harmonic;
        for (source_idx, &source) in therapy_elements.iter().enumerate() {
            for &offset in settings.receiver_offsets {
                let receiver_idx = (source_idx + offset).rem_euclid(therapy_elements.len().max(1));
                rows.push(PitchCatchRow {
                    source,
                    receiver: therapy_elements[receiver_idx],
                    k,
                    frequency_mhz,
                    nonlinear_path_weight,
                });
            }
        }
    }
    rows
}

pub(super) fn passive_rows(
    therapy_elements: &[PlanarPoint],
    imaging_receivers: &[PlanarPoint],
    frequencies_hz: &[f64],
) -> Vec<PassiveRow> {
    let receiver_count = therapy_elements.len() + imaging_receivers.len();
    let mut rows = Vec::with_capacity(2 * receiver_count * frequencies_hz.len());
    for &frequency_hz in frequencies_hz {
        let k = std::f64::consts::TAU * (0.5 * frequency_hz) / C_REF_M_S;
        let frequency_mhz = 0.5 * frequency_hz * 1.0e-6;
        for (receiver_idx, &receiver) in therapy_elements
            .iter()
            .chain(imaging_receivers.iter())
            .enumerate()
        {
            let _ = receiver_idx;
            for sine_phase in [false, true] {
                rows.push(PassiveRow {
                    receiver,
                    k,
                    frequency_mhz,
                    sine_phase,
                });
            }
        }
    }
    rows
}

pub(super) fn compute_row_norms(
    kind: &OperatorKind,
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) -> Vec<f32> {
    match kind {
        OperatorKind::PitchCatch(specs) => specs
            .par_iter()
            .map(|spec| pitch_catch_row_norm(spec, active, medium))
            .collect(),
        OperatorKind::Passive(specs) => specs
            .par_iter()
            .map(|spec| passive_row_norm(spec, active, medium))
            .collect(),
    }
}

#[inline]
fn pitch_catch_row_norm(
    spec: &PitchCatchRow,
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) -> f32 {
    let mut norm_sq = 0.0_f32;
    for col in 0..active.len() {
        let (point, alpha) = column_lookup(active, medium, col);
        let value = spec.unscaled_value(point, alpha, medium.spacing_m);
        norm_sq += value * value;
    }
    norm_sq.sqrt().max(f32::EPSILON)
}

#[inline]
fn passive_row_norm(spec: &PassiveRow, active: &ActiveGrid, medium: SameApertureMedium<'_>) -> f32 {
    let mut norm_sq = 0.0_f32;
    for col in 0..active.len() {
        let (point, alpha) = column_lookup(active, medium, col);
        let value = spec.unscaled_value(point, alpha, medium.spacing_m);
        norm_sq += value * value;
    }
    norm_sq.sqrt().max(f32::EPSILON)
}

#[inline]
pub(super) fn write_pitch_catch_row(
    row: &mut [f32],
    spec: &PitchCatchRow,
    inv_norm: f32,
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) {
    for (col, dst) in row.iter_mut().enumerate() {
        let (point, alpha) = column_lookup(active, medium, col);
        *dst = spec.unscaled_value(point, alpha, medium.spacing_m) * inv_norm;
    }
}

#[inline]
pub(super) fn write_passive_row(
    row: &mut [f32],
    spec: &PassiveRow,
    inv_norm: f32,
    active: &ActiveGrid,
    medium: SameApertureMedium<'_>,
) {
    for (col, dst) in row.iter_mut().enumerate() {
        let (point, alpha) = column_lookup(active, medium, col);
        *dst = spec.unscaled_value(point, alpha, medium.spacing_m) * inv_norm;
    }
}
