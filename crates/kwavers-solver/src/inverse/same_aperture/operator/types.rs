//! Operator struct + row-spec value types.

use super::super::active_grid::{ActiveGrid, PlanarPoint};
use super::super::finite_frequency::{SameApertureMedium, SameApertureSettings};
use super::super::row_matrix::RowMatrix;
use super::dot::distance;
use super::rows::{
    compute_row_norms, passive_rows, pitch_catch_rows, write_passive_row, write_pitch_catch_row,
};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct FiniteFrequencyOperator<'a> {
    pub(super) medium: SameApertureMedium<'a>,
    pub(super) active: &'a ActiveGrid,
    pub(super) kind: OperatorKind,
    pub(super) rows: usize,
    pub(super) row_norms: Vec<f32>,
    pub(super) inv_row_norms: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(super) enum OperatorKind {
    PitchCatch(Vec<PitchCatchRow>),
    Passive(Vec<PassiveRow>),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PitchCatchRow {
    pub(super) source: PlanarPoint,
    pub(super) receiver: PlanarPoint,
    pub(super) k: f64,
    pub(super) frequency_mhz: f64,
    pub(super) nonlinear_path_weight: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PassiveRow {
    pub(super) receiver: PlanarPoint,
    pub(super) k: f64,
    pub(super) frequency_mhz: f64,
    pub(super) sine_phase: bool,
}

impl PitchCatchRow {
    #[inline(always)]
    pub(super) fn unscaled_value(&self, point: PlanarPoint, alpha: f64, spacing_m: f64) -> f32 {
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
    pub(super) fn unscaled_value(&self, point: PlanarPoint, alpha: f64, spacing_m: f64) -> f32 {
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
