//! Matrix-free finite-frequency same-aperture operators.
//!
//! # Performance contract
//!
//! `FiniteFrequencyOperator` precomputes one `PitchCatchRow` or `PassiveRow`
//! record per output row at construction so the hot `matvec`, `t_matvec`,
//! `normal_diag`, and `materialize` loops never recompute the row index
//! `divmod`, the source/receiver pair, the angular wavenumber, or the
//! frequency-MHz factor on a per-cell basis. Inverse row norms are cached
//! alongside the row norms so the inner loops never recompute `1 / norm`.
//! Outer loops over rows or columns dispatch through rayon for cache-aware
//! parallelism on the SPD normal equations driven by PCG.

use rayon::prelude::*;

use super::active_grid::{ActiveGrid, PlanarPoint};
use super::finite_frequency::{SameApertureMedium, SameApertureSettings, C_REF_M_S};
use super::linear_operator::LinearOperator;
use super::row_matrix::RowMatrix;

#[derive(Clone, Debug)]
pub struct FiniteFrequencyOperator<'a> {
    medium: SameApertureMedium<'a>,
    active: &'a ActiveGrid,
    kind: OperatorKind,
    rows: usize,
    row_norms: Vec<f32>,
    inv_row_norms: Vec<f32>,
}

#[derive(Clone, Debug)]
enum OperatorKind {
    PitchCatch(Vec<PitchCatchRow>),
    Passive(Vec<PassiveRow>),
}

#[derive(Clone, Copy, Debug)]
struct PitchCatchRow {
    source: PlanarPoint,
    receiver: PlanarPoint,
    k: f64,
    frequency_mhz: f64,
    nonlinear_path_weight: bool,
}

#[derive(Clone, Copy, Debug)]
struct PassiveRow {
    receiver: PlanarPoint,
    k: f64,
    frequency_mhz: f64,
    sine_phase: bool,
}

impl PitchCatchRow {
    #[inline(always)]
    fn unscaled_value(&self, point: PlanarPoint, alpha: f64, spacing_m: f64) -> f32 {
        let ds = distance(point, self.source).max(spacing_m);
        let dr = distance(point, self.receiver).max(spacing_m);
        let path_m = ds + dr;
        let attenuation = (-alpha * self.frequency_mhz * path_m).exp();
        let weight = if self.nonlinear_path_weight {
            path_m
        } else {
            1.0
        };
        (spacing_m * spacing_m * weight * attenuation * (self.k * path_m).cos() / (ds * dr).sqrt())
            as f32
    }
}

impl PassiveRow {
    #[inline(always)]
    fn unscaled_value(&self, point: PlanarPoint, alpha: f64, spacing_m: f64) -> f32 {
        let dr = distance(point, self.receiver).max(spacing_m);
        let phase = if self.sine_phase {
            (self.k * dr).sin()
        } else {
            (self.k * dr).cos()
        };
        (spacing_m * spacing_m * (-alpha * self.frequency_mhz * dr).exp() * phase / dr.sqrt())
            as f32
    }
}

impl<'a> FiniteFrequencyOperator<'a> {
    #[must_use]
    pub fn fundamental(
        medium: SameApertureMedium<'a>,
        therapy_elements: &[PlanarPoint],
        active: &'a ActiveGrid,
        settings: SameApertureSettings<'_>,
    ) -> Self {
        Self::pitch_catch(medium, therapy_elements, active, settings, 1.0)
    }

    #[must_use]
    pub fn harmonic(
        medium: SameApertureMedium<'a>,
        therapy_elements: &[PlanarPoint],
        active: &'a ActiveGrid,
        settings: SameApertureSettings<'_>,
    ) -> Self {
        Self::pitch_catch(medium, therapy_elements, active, settings, 2.0)
    }

    #[must_use]
    pub fn ultraharmonic(
        medium: SameApertureMedium<'a>,
        therapy_elements: &[PlanarPoint],
        active: &'a ActiveGrid,
        settings: SameApertureSettings<'_>,
    ) -> Self {
        Self::pitch_catch(medium, therapy_elements, active, settings, 1.5)
    }

    #[must_use]
    pub fn passive(
        medium: SameApertureMedium<'a>,
        therapy_elements: &[PlanarPoint],
        imaging_receivers: &[PlanarPoint],
        active: &'a ActiveGrid,
        frequencies_hz: &[f64],
    ) -> Self {
        let rows = passive_rows(therapy_elements, imaging_receivers, frequencies_hz);
        Self::new(medium, active, OperatorKind::Passive(rows))
    }

    #[must_use]
    pub fn materialize(&self) -> RowMatrix {
        let mut matrix = RowMatrix::zeros(self.rows, self.active.len());
        let cols = self.active.len();
        match &self.kind {
            OperatorKind::PitchCatch(specs) => {
                matrix
                    .data
                    .par_chunks_mut(cols)
                    .zip(specs.par_iter())
                    .zip(self.inv_row_norms.par_iter())
                    .for_each(|((row_slice, spec), inv_norm)| {
                        write_pitch_catch_row(row_slice, spec, *inv_norm, self.active, self.medium);
                    });
            }
            OperatorKind::Passive(specs) => {
                matrix
                    .data
                    .par_chunks_mut(cols)
                    .zip(specs.par_iter())
                    .zip(self.inv_row_norms.par_iter())
                    .for_each(|((row_slice, spec), inv_norm)| {
                        write_passive_row(row_slice, spec, *inv_norm, self.active, self.medium);
                    });
            }
        }
        matrix
    }

    fn pitch_catch(
        medium: SameApertureMedium<'a>,
        therapy_elements: &[PlanarPoint],
        active: &'a ActiveGrid,
        settings: SameApertureSettings<'_>,
        harmonic: f64,
    ) -> Self {
        let rows = pitch_catch_rows(therapy_elements, settings, harmonic);
        Self::new(medium, active, OperatorKind::PitchCatch(rows))
    }

    fn new(medium: SameApertureMedium<'a>, active: &'a ActiveGrid, kind: OperatorKind) -> Self {
        let rows = match &kind {
            OperatorKind::PitchCatch(specs) => specs.len(),
            OperatorKind::Passive(specs) => specs.len(),
        };
        let row_norms = compute_row_norms(&kind, active, medium);
        let inv_row_norms = row_norms.iter().map(|n| 1.0 / *n).collect();
        Self {
            medium,
            active,
            kind,
            rows,
            row_norms,
            inv_row_norms,
        }
    }
}

impl LinearOperator for FiniteFrequencyOperator<'_> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.active.len()
    }

    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols());
        debug_assert_eq!(out.len(), self.rows());
        match &self.kind {
            OperatorKind::PitchCatch(specs) => {
                out.par_iter_mut()
                    .zip(specs.par_iter())
                    .zip(self.inv_row_norms.par_iter())
                    .for_each(|((dst, spec), inv_norm)| {
                        *dst = pitch_catch_row_dot(spec, *inv_norm, x, self.active, self.medium);
                    });
            }
            OperatorKind::Passive(specs) => {
                out.par_iter_mut()
                    .zip(specs.par_iter())
                    .zip(self.inv_row_norms.par_iter())
                    .for_each(|((dst, spec), inv_norm)| {
                        *dst = passive_row_dot(spec, *inv_norm, x, self.active, self.medium);
                    });
            }
        }
    }

    fn t_matvec(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.rows());
        debug_assert_eq!(out.len(), self.cols());
        let scaled = scaled_input(y, &self.inv_row_norms);
        match &self.kind {
            OperatorKind::PitchCatch(specs) => {
                out.par_iter_mut().enumerate().for_each(|(col, dst)| {
                    *dst = pitch_catch_column_dot(specs, &scaled, col, self.active, self.medium);
                });
            }
            OperatorKind::Passive(specs) => {
                out.par_iter_mut().enumerate().for_each(|(col, dst)| {
                    *dst = passive_column_dot(specs, &scaled, col, self.active, self.medium);
                });
            }
        }
    }

    fn row_values(&self, row: usize, out: &mut [f32]) {
        debug_assert!(row < self.rows());
        debug_assert_eq!(out.len(), self.cols());
        match &self.kind {
            OperatorKind::PitchCatch(specs) => {
                write_pitch_catch_row(
                    out,
                    &specs[row],
                    self.inv_row_norms[row],
                    self.active,
                    self.medium,
                );
            }
            OperatorKind::Passive(specs) => {
                write_passive_row(
                    out,
                    &specs[row],
                    self.inv_row_norms[row],
                    self.active,
                    self.medium,
                );
            }
        }
    }

    fn normal_diag(&self) -> Vec<f32> {
        let mut diag = vec![0.0_f32; self.cols()];
        match &self.kind {
            OperatorKind::PitchCatch(specs) => {
                diag.par_iter_mut().enumerate().for_each(|(col, dst)| {
                    *dst = pitch_catch_column_norm_sq(
                        specs,
                        &self.inv_row_norms,
                        col,
                        self.active,
                        self.medium,
                    );
                });
            }
            OperatorKind::Passive(specs) => {
                diag.par_iter_mut().enumerate().for_each(|(col, dst)| {
                    *dst = passive_column_norm_sq(
                        specs,
                        &self.inv_row_norms,
                        col,
                        self.active,
                        self.medium,
                    );
                });
            }
        }
        diag
    }

    fn storage_values(&self) -> usize {
        let kind_values = match &self.kind {
            OperatorKind::PitchCatch(specs) => specs.len() * PITCH_CATCH_ROW_VALUES,
            OperatorKind::Passive(specs) => specs.len() * PASSIVE_ROW_VALUES,
        };
        self.row_norms.len() + self.inv_row_norms.len() + kind_values
    }
}

const PITCH_CATCH_ROW_VALUES: usize = 7;
const PASSIVE_ROW_VALUES: usize = 5;

fn pitch_catch_rows(
    therapy_elements: &[PlanarPoint],
    settings: SameApertureSettings<'_>,
    harmonic: f64,
) -> Vec<PitchCatchRow> {
    let nonlinear_path_weight = harmonic > 1.0;
    let mut rows = Vec::with_capacity(
        settings.frequencies_hz.len() * settings.receiver_offsets.len() * therapy_elements.len(),
    );
    for &frequency_hz in settings.frequencies_hz {
        let k = std::f64::consts::TAU * frequency_hz * harmonic / C_REF_M_S;
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

fn passive_rows(
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

fn compute_row_norms(
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
fn pitch_catch_row_dot(
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
fn passive_row_dot(
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
fn pitch_catch_column_dot(
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
fn passive_column_dot(
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
fn pitch_catch_column_norm_sq(
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
fn passive_column_norm_sq(
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
fn write_pitch_catch_row(
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
fn write_passive_row(
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

#[inline]
fn column_lookup(
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
fn scaled_input(y: &[f32], inv_row_norms: &[f32]) -> Vec<f32> {
    y.iter()
        .zip(inv_row_norms.iter())
        .map(|(yv, inv)| yv * inv)
        .collect()
}

#[inline]
fn distance(a: PlanarPoint, b: PlanarPoint) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}
