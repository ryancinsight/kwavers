//! Dense matrix kernels for linear Born inversion.
//!
//! These kernels operate on a row-major sensitivity matrix `A` with `nrows`
//! measurements and `ncols` unknowns. They are anatomy-neutral: clinical
//! adapters own geometry, media, and active-voxel extraction, then pass the
//! assembled dense operator here.

use moirai_parallel::{fold_reduce_with, reduce_index_with, Adaptive};

use super::LinearBornInversionConfig;

/// Diagonal-preconditioned migration image `diag(A^T A + λI)^-1 A^T b`.
pub fn migration_contrast(
    matrix: &[f64],
    data: &[f64],
    nrows: usize,
    ncols: usize,
    config: &LinearBornInversionConfig,
) -> Vec<f64> {
    let diag = normal_equation_diagonal(matrix, nrows, ncols, config.regularization);
    let rows: Vec<usize> = (0..nrows).collect();
    let mut adjoint = adjoint_rows(matrix, data, &rows, ncols);
    for col in 0..ncols {
        adjoint[col] = (adjoint[col] / diag[col]).clamp(config.contrast_min, config.contrast_max);
    }
    adjoint
}

/// Compute `diag(A^T A) + λI` over all rows.
pub(crate) fn normal_equation_diagonal(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    regularization: f64,
) -> Vec<f64> {
    let rows: Vec<usize> = (0..nrows).collect();
    normal_equation_diagonal_rows(matrix, &rows, ncols, regularization)
}

/// Compute `diag(A[rows]^T A[rows]) + λI`.
pub fn normal_equation_diagonal_rows(
    matrix: &[f64],
    rows: &[usize],
    ncols: usize,
    regularization: f64,
) -> Vec<f64> {
    let mut diagonal = fold_reduce_with::<Adaptive, _, _, _, _>(
        rows.len(),
        || vec![0.0f64; ncols],
        |mut partial, row_index| {
            let row = rows[row_index];
            let base = row * ncols;
            for col in 0..ncols {
                let a = matrix[base + col];
                partial[col] += a * a;
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
    let reg = regularization.max(1.0e-12);
    for d in &mut diagonal {
        *d += reg;
    }
    diagonal
}

/// Compute diagonal-preconditioned `A^T(b - A model) - λ model` over selected rows.
pub fn normalized_gradient_rows(
    matrix: &[f64],
    data: &[f64],
    model: &[f64],
    rows: &[usize],
    ncols: usize,
    diag: &[f64],
    config: &LinearBornInversionConfig,
) -> Vec<f64> {
    let mut gradient = adjoint_residual_rows(matrix, data, model, rows, ncols);
    for col in 0..ncols {
        gradient[col] = (gradient[col] - config.regularization * model[col]) / diag[col];
    }
    gradient
}

/// Compute `A x` with `x` supplied by index.
pub fn matrix_vector<F>(matrix: &[f64], nrows: usize, ncols: usize, x: F) -> Vec<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    let mut data = vec![0.0; nrows];
    for (row, value) in data.iter_mut().enumerate() {
        let base = row * ncols;
        let mut acc = 0.0;
        for col in 0..ncols {
            acc += matrix[base + col] * x(col);
        }
        *value = acc;
    }
    data
}

/// Half-squared L2 data misfit plus Tikhonov regularization over all rows.
pub fn objective(
    matrix: &[f64],
    data: &[f64],
    model: &[f64],
    nrows: usize,
    ncols: usize,
    regularization: f64,
) -> f64 {
    let rows: Vec<usize> = (0..nrows).collect();
    objective_rows(matrix, data, model, &rows, ncols, regularization)
}

/// Half-squared L2 data misfit plus Tikhonov regularization over selected rows.
pub fn objective_rows(
    matrix: &[f64],
    data: &[f64],
    model: &[f64],
    rows: &[usize],
    ncols: usize,
    regularization: f64,
) -> f64 {
    let misfit = reduce_index_with::<Adaptive, _, _, _>(
        rows.len(),
        0.0,
        |row_index| {
            let row = rows[row_index];
            let datum = data[row];
            let base = row * ncols;
            let mut pred = 0.0;
            for col in 0..ncols {
                pred += matrix[base + col] * model[col];
            }
            let residual = datum - pred;
            0.5 * residual * residual
        },
        |a, b| a + b,
    );
    misfit + 0.5 * regularization * model.iter().map(|v| v * v).sum::<f64>()
}

fn adjoint_rows(matrix: &[f64], data: &[f64], rows: &[usize], ncols: usize) -> Vec<f64> {
    fold_reduce_with::<Adaptive, _, _, _, _>(
        rows.len(),
        || vec![0.0f64; ncols],
        |mut partial, row_index| {
            let row = rows[row_index];
            let datum = data[row];
            let base = row * ncols;
            for col in 0..ncols {
                partial[col] += matrix[base + col] * datum;
            }
            partial
        },
        |mut a, b| {
            for (ai, bi) in a.iter_mut().zip(&b) {
                *ai += bi;
            }
            a
        },
    )
}

fn adjoint_residual_rows(
    matrix: &[f64],
    data: &[f64],
    model: &[f64],
    rows: &[usize],
    ncols: usize,
) -> Vec<f64> {
    fold_reduce_with::<Adaptive, _, _, _, _>(
        rows.len(),
        || vec![0.0f64; ncols],
        |mut partial, row_index| {
            let row = rows[row_index];
            let datum = data[row];
            let base = row * ncols;
            let mut pred = 0.0;
            for col in 0..ncols {
                pred += matrix[base + col] * model[col];
            }
            let residual = datum - pred;
            for col in 0..ncols {
                partial[col] += matrix[base + col] * residual;
            }
            partial
        },
        |mut a, b| {
            for (ai, bi) in a.iter_mut().zip(&b) {
                *ai += bi;
            }
            a
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_objective_and_gradient_match_hand_computation() {
        let matrix = [1.0, 2.0, -1.0, 0.5, 0.0, 3.0];
        let data = matrix_vector(&matrix, 3, 2, |col| [0.25, -0.5][col]);
        assert_eq!(data, vec![-0.75, -0.5, -1.5]);

        let objective_value = objective(&matrix, &data, &[0.0, 0.0], 3, 2, 0.2);
        let expected = 0.5 * (0.75_f64.powi(2) + 0.5_f64.powi(2) + 1.5_f64.powi(2));
        assert!((objective_value - expected).abs() < 1.0e-12);

        let config = LinearBornInversionConfig {
            regularization: 0.2,
            ..LinearBornInversionConfig::default()
        };
        let rows = [0, 1, 2];
        let diag = normal_equation_diagonal_rows(&matrix, &rows, 2, config.regularization);
        assert!((diag[0] - 2.2).abs() < 1.0e-12);
        assert!((diag[1] - 13.45).abs() < 1.0e-12);

        let gradient =
            normalized_gradient_rows(&matrix, &data, &[0.0, 0.0], &rows, 2, &diag, &config);
        assert!((gradient[0] + 0.11363636363636363).abs() < 1.0e-12);
        assert!((gradient[1] + 0.4646840148698885).abs() < 1.0e-12);
    }
}
