//! Parallel operators for [`super::VolumeOperator`]: forward projection, adjoint,
//! objective, normal-equation residual, normal matrix-vector product, and
//! private row-evaluation helpers.

use crate::inverse::linear_born_inversion::LinearBornInversionConfig;
use moirai_parallel::{fold_reduce_with, map_collect_index_with, reduce_index_with, Adaptive};

use super::{RowContext, VolumeOperator};

impl<'a> VolumeOperator<'a> {
    /// Compute the L2 row-normalisation factors `‖A[row, :]‖` for all rows.
    pub fn row_norms(&self) -> Vec<f64> {
        map_collect_index_with::<Adaptive, _, _>(self.row_contexts.len(), |row| self.row_norm(row))
    }

    /// Synthetic data vector `A · target_contrast` for all rows.
    pub fn data_from_target(&self, row_norms: &[f64]) -> Vec<f64> {
        map_collect_index_with::<Adaptive, _, _>(self.row_contexts.len(), |row| {
            self.project_row_with_norm(row, row_norms[row], |col| self.active[col].target_contrast)
        })
    }

    /// Compute the normal-equation diagonal: `diag(AᵀA) + regularization`.
    ///
    /// Uses `fold + reduce` so each Moirai worker shard accumulates into a
    /// task-local partial Vec, then Moirai combines partials; no intermediate
    /// `Vec<Vec<f64>>` is collected before reduction.
    pub fn diagonal(
        &self,
        rows: &[usize],
        row_norms: &[f64],
        config: &LinearBornInversionConfig,
    ) -> Vec<f64> {
        let ncols = self.n_active;
        let reg = config.regularization.max(1.0e-12);
        let mut diagonal = fold_reduce_with::<Adaptive, _, _, _, _>(
            rows.len(),
            || vec![0.0f64; ncols],
            |mut partial, row_index| {
                let row = rows[row_index];
                let norm = row_norms[row];
                if norm != 0.0 {
                    let row_context = self.row_context(row);
                    for (col, p) in partial.iter_mut().enumerate() {
                        let value = self.row_value_for_col(&row_context, col) / norm;
                        *p += value * value;
                    }
                }
                partial
            },
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(&b) {
                    *ai += bi;
                }
                a
            },
        );
        // Add regularization once after reduction; avoids counting it n_tasks times.
        for d in diagonal.iter_mut() {
            *d += reg;
        }
        diagonal
    }

    /// Diagonal-preconditioned adjoint migration image `diag(AᵀA)⁻¹ Aᵀ b`.
    pub fn migration(
        &self,
        data: &[f64],
        rows: &[usize],
        row_norms: &[f64],
        config: &LinearBornInversionConfig,
    ) -> Vec<f64> {
        let ncols = self.n_active;
        let diagonal = self.diagonal(rows, row_norms, config);
        let adjoint = fold_reduce_with::<Adaptive, _, _, _, _>(
            rows.len(),
            || vec![0.0f64; ncols],
            |mut partial, row_index| {
                let row = rows[row_index];
                let norm = row_norms[row];
                if norm != 0.0 {
                    let row_context = self.row_context(row);
                    for (col, p) in partial.iter_mut().enumerate() {
                        *p += self.row_value_for_col(&row_context, col) * data[row] / norm;
                    }
                }
                partial
            },
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(&b) {
                    *ai += bi;
                }
                a
            },
        );
        adjoint
            .into_iter()
            .zip(diagonal)
            .map(|(value, diag)| (value / diag).clamp(config.contrast_min, config.contrast_max))
            .collect()
    }

    /// Half-squared L2 data misfit plus Tikhonov regularisation.
    pub fn objective(
        &self,
        data: &[f64],
        model: &[f64],
        rows: &[usize],
        row_norms: &[f64],
        regularization: f64,
    ) -> f64 {
        let data_misfit = reduce_index_with::<Adaptive, _, _, _>(
            rows.len(),
            0.0,
            |row_index| {
                let row = rows[row_index];
                let prediction = self.project_row_with_norm(row, row_norms[row], |col| model[col]);
                let residual = data[row] - prediction;
                0.5 * residual * residual
            },
            |a, b| a + b,
        );
        data_misfit + 0.5 * regularization * model.iter().map(|v| v * v).sum::<f64>()
    }

    /// Compute `AᵀA·x − Aᵀb` (normal-equation residual).
    ///
    /// Fold state `(partial, row_values)` amortises the `row_values` buffer
    /// across rows within each task: one allocation per task instead of one per
    /// row.  The `row_values` Vec is dropped after `map(|(p, _)| p)` before the
    /// tree reduce step.
    pub fn normal_residual(
        &self,
        data: &[f64],
        model: &[f64],
        rows: &[usize],
        row_norms: &[f64],
        regularization: f64,
    ) -> Vec<f64> {
        let ncols = self.n_active;
        let mut residual = fold_reduce_with::<Adaptive, _, _, _, _>(
            rows.len(),
            || (vec![0.0f64; ncols], vec![0.0f64; ncols]),
            |(mut partial, mut row_values), row_index| {
                let row = rows[row_index];
                let norm = row_norms[row];
                if norm != 0.0 {
                    self.fill_row_values(row, &mut row_values);
                    let prediction: f64 = row_values
                        .iter()
                        .zip(model)
                        .map(|(rv, mv)| rv * mv / norm)
                        .sum();
                    let row_residual = data[row] - prediction;
                    for (pv, rv) in partial.iter_mut().zip(&row_values) {
                        *pv += rv * row_residual / norm;
                    }
                }
                (partial, row_values)
            },
            |mut a, b| {
                for (ai, bi) in a.0.iter_mut().zip(&b.0) {
                    *ai += bi;
                }
                a
            },
        )
        .0;
        for (value, model_value) in residual.iter_mut().zip(model) {
            *value -= regularization * model_value;
        }
        residual
    }

    /// Compute `(AᵀA + λI)·p` (normal-equation matrix-vector product).
    ///
    /// Fold state `(partial, row_values)` amortises the row buffer as in
    /// `normal_residual`.
    pub fn apply_normal(
        &self,
        vector: &[f64],
        rows: &[usize],
        row_norms: &[f64],
        regularization: f64,
    ) -> Vec<f64> {
        let ncols = self.n_active;
        let mut out = fold_reduce_with::<Adaptive, _, _, _, _>(
            rows.len(),
            || (vec![0.0f64; ncols], vec![0.0f64; ncols]),
            |(mut partial, mut row_values), row_index| {
                let row = rows[row_index];
                let norm = row_norms[row];
                if norm != 0.0 {
                    self.fill_row_values(row, &mut row_values);
                    let projection: f64 = row_values
                        .iter()
                        .zip(vector)
                        .map(|(rv, vv)| rv * vv / norm)
                        .sum();
                    for (pv, rv) in partial.iter_mut().zip(&row_values) {
                        *pv += rv * projection / norm;
                    }
                }
                (partial, row_values)
            },
            |mut a, b| {
                for (ai, bi) in a.0.iter_mut().zip(&b.0) {
                    *ai += bi;
                }
                a
            },
        )
        .0;
        for (value, vector_value) in out.iter_mut().zip(vector) {
            *value += regularization * vector_value;
        }
        out
    }

    fn project_row_with_norm<F>(&self, row: usize, norm: f64, value_at: F) -> f64
    where
        F: Fn(usize) -> f64 + Sync,
    {
        if norm == 0.0 {
            return 0.0;
        }
        let row_context = self.row_context(row);
        (0..self.n_active)
            .map(|col| self.row_value_for_col(&row_context, col) * value_at(col) / norm)
            .sum()
    }

    fn row_norm(&self, row: usize) -> f64 {
        let row_context = self.row_context(row);
        (0..self.n_active)
            .map(|col| {
                let v = self.row_value_for_col(&row_context, col);
                v * v
            })
            .sum::<f64>()
            .sqrt()
    }

    fn fill_row_values(&self, row: usize, out: &mut [f64]) {
        let row_context = self.row_context(row);
        for (col, value) in out.iter_mut().enumerate() {
            *value = self.row_value_for_col(&row_context, col);
        }
    }

    fn row_context(&self, row: usize) -> RowContext {
        self.row_contexts[row]
    }
}
