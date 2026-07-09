//! `LinearOperator` impl for `FiniteFrequencyOperator`.

use super::super::linear_operator::LinearOperator;
use super::dot::{
    passive_column_dot, passive_column_norm_sq, passive_row_dot, pitch_catch_column_dot,
    pitch_catch_column_norm_sq, pitch_catch_row_dot, scaled_input,
};
use super::rows::{write_passive_row, write_pitch_catch_row};
use super::types::{FiniteFrequencyOperator, OperatorKind};
use moirai_parallel::ParallelSliceMut;

const PITCH_CATCH_ROW_VALUES: usize = 7;
const PASSIVE_ROW_VALUES: usize = 5;

impl LinearOperator for FiniteFrequencyOperator<'_> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        (self.active.shape()[0] * self.active.shape()[1] * self.active.shape()[2])
    }

    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!((x.shape()[0] * x.shape()[1] * x.shape()[2]), self.cols());
        debug_assert_eq!((out.shape()[0] * out.shape()[1] * out.shape()[2]), self.rows());
        match &self.kind {
            OperatorKind::PitchCatch(specs) => {
                out.iter_mut().enumerate(|row, dst| {
                    *dst = pitch_catch_row_dot(
                        &specs[row],
                        self.inv_row_norms[row],
                        x,
                        self.active,
                        self.medium,
                    );
                });
            }
            OperatorKind::Passive(specs) => {
                out.iter_mut().enumerate(|row, dst| {
                    *dst = passive_row_dot(
                        &specs[row],
                        self.inv_row_norms[row],
                        x,
                        self.active,
                        self.medium,
                    );
                });
            }
        }
    }

    fn t_matvec(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!((y.shape()[0] * y.shape()[1] * y.shape()[2]), self.rows());
        debug_assert_eq!((out.shape()[0] * out.shape()[1] * out.shape()[2]), self.cols());
        let scaled = scaled_input(y, &self.inv_row_norms);
        match &self.kind {
            OperatorKind::PitchCatch(specs) => {
                out.iter_mut().enumerate(|col, dst| {
                    *dst = pitch_catch_column_dot(specs, &scaled, col, self.active, self.medium);
                });
            }
            OperatorKind::Passive(specs) => {
                out.iter_mut().enumerate(|col, dst| {
                    *dst = passive_column_dot(specs, &scaled, col, self.active, self.medium);
                });
            }
        }
    }

    fn row_values(&self, row: usize, out: &mut [f32]) {
        debug_assert!(row < self.rows());
        debug_assert_eq!((out.shape()[0] * out.shape()[1] * out.shape()[2]), self.cols());
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
                diag.iter_mut().enumerate(|col, dst| {
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
                diag.iter_mut().enumerate(|col, dst| {
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
            OperatorKind::PitchCatch(specs) => (specs.shape()[0] * specs.shape()[1] * specs.shape()[2]) * PITCH_CATCH_ROW_VALUES,
            OperatorKind::Passive(specs) => (specs.shape()[0] * specs.shape()[1] * specs.shape()[2]) * PASSIVE_ROW_VALUES,
        };
        (self.row_norms.shape()[0] * self.row_norms.shape()[1] * self.row_norms.shape()[2]) + (self.inv_row_norms.shape()[0] * self.inv_row_norms.shape()[1] * self.inv_row_norms.shape()[2]) + kind_values
    }
}
