//! Matrix-vector algebra over the speed-shift row storage.
//!
//! ## Parallelism invariants
//!
//! `matvec` (A·x → y): each output element `y[row]` depends only on one row of
//! A and is written exactly once → embarrassingly parallel with
//! indexed mutable Moirai dispatch.
//!
//! `t_matvec` (Aᵀ·y → x) and `normal_diag_into`: scatter patterns where
//! multiple rows contribute to the same column.  Both use the `fold + reduce`
//! pattern: each Moirai task accumulates into a task-local partial vector; binary
//! tree reduction combines partials without atomics.
//!
//! Race-freedom: `fold` identity `vec![0.0; ncols]` is cloned once per task;
//! `reduce` identity is also `vec![0.0; ncols]`.  No two tasks share a partial
//! buffer, so writes to `partial[col]` are thread-private.

use moirai_parallel::{enumerate_mut_with, fold_reduce_with, Adaptive};

use super::SoundSpeedShiftOperator;

impl SoundSpeedShiftOperator {
    pub(in crate::reconstruction::sound_speed_shift) fn matvec(&self, x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(x.len(), self.cols());
        debug_assert_eq!(out.len(), self.rows());
        // Each row is independent: y[row] = Σ_{col} A[row,col]·x[col]
        enumerate_mut_with::<Adaptive, _, _>(out, |row, dst| {
            *dst = self
                .rows
                .row_entries(row)
                .map(|(col, length)| length * x[col])
                .sum();
        });
    }

    pub(in crate::reconstruction::sound_speed_shift) fn t_matvec(
        &self,
        y: &[f64],
        out: &mut [f64],
    ) {
        debug_assert_eq!(y.len(), self.rows());
        debug_assert_eq!(out.len(), self.cols());
        let ncols = self.cols();
        let nrows = self.rows();
        // fold + reduce: task-local partials → binary-tree combine.
        let result = fold_reduce_with::<Adaptive, Vec<f64>, _, _, _>(
            nrows,
            || vec![0.0f64; ncols],
            |mut partial, row| {
                let yv = y[row];
                for (col, length) in self.rows.row_entries(row) {
                    partial[col] += length * yv;
                }
                partial
            },
            sum_partials,
        );
        out.copy_from_slice(&result);
    }

    pub(in crate::reconstruction::sound_speed_shift) fn normal_diag_into(&self, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.cols());
        let ncols = self.cols();
        let nrows = self.rows();
        let result = fold_reduce_with::<Adaptive, Vec<f64>, _, _, _>(
            nrows,
            || vec![0.0f64; ncols],
            |mut partial, row| {
                for (col, length) in self.rows.row_entries(row) {
                    partial[col] += length * length;
                }
                partial
            },
            sum_partials,
        );
        out.copy_from_slice(&result);
    }
}

fn sum_partials(mut a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
    a
}
