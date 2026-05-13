//! Dense finite-frequency inverse-problem kernels.

use rayon::prelude::*;

use super::config::BrainHelmetFwiConfig;

pub(super) fn migration_contrast(
    matrix: &[f64],
    data: &[f64],
    nrows: usize,
    ncols: usize,
    config: &BrainHelmetFwiConfig,
) -> Vec<f64> {
    let diag = normal_equation_diagonal(matrix, nrows, ncols, config.regularization);
    let rows: Vec<usize> = (0..nrows).collect();
    let mut adjoint = adjoint_rows(matrix, data, &rows, ncols);
    for col in 0..ncols {
        adjoint[col] = (adjoint[col] / diag[col]).clamp(config.contrast_min, config.contrast_max);
    }
    adjoint
}

pub(super) fn normal_equation_diagonal(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    regularization: f64,
) -> Vec<f64> {
    let rows: Vec<usize> = (0..nrows).collect();
    normal_equation_diagonal_rows(matrix, &rows, ncols, regularization)
}

pub(super) fn normal_equation_diagonal_rows(
    matrix: &[f64],
    rows: &[usize],
    ncols: usize,
    regularization: f64,
) -> Vec<f64> {
    let mut diag = vec![regularization.max(1.0e-12); ncols];
    let partials: Vec<Vec<f64>> = rows
        .par_chunks(row_chunk_len(rows.len()))
        .map(|chunk| {
            let mut partial = vec![0.0; ncols];
            for &row in chunk {
                let base = row * ncols;
                for col in 0..ncols {
                    let a = matrix[base + col];
                    partial[col] += a * a;
                }
            }
            partial
        })
        .collect();
    for partial in partials {
        add_vector_in_place(&mut diag, &partial);
    }
    diag
}

pub(super) fn normalized_gradient_rows(
    matrix: &[f64],
    data: &[f64],
    model: &[f64],
    rows: &[usize],
    ncols: usize,
    diag: &[f64],
    config: &BrainHelmetFwiConfig,
) -> Vec<f64> {
    let mut gradient = adjoint_residual_rows(matrix, data, model, rows, ncols);
    for col in 0..ncols {
        gradient[col] = (gradient[col] - config.regularization * model[col]) / diag[col];
    }
    gradient
}

pub(super) fn matrix_vector<F>(matrix: &[f64], nrows: usize, ncols: usize, x: F) -> Vec<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    let mut data = vec![0.0; nrows];
    data.par_iter_mut()
        .enumerate()
        .take(nrows)
        .for_each(|(row, value)| {
            let base = row * ncols;
            let mut acc = 0.0;
            for col in 0..ncols {
                acc += matrix[base + col] * x(col);
            }
            *value = acc;
        });
    data
}

pub(super) fn objective(
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

pub(super) fn objective_rows(
    matrix: &[f64],
    data: &[f64],
    model: &[f64],
    rows: &[usize],
    ncols: usize,
    regularization: f64,
) -> f64 {
    let partials: Vec<f64> = rows
        .par_chunks(row_chunk_len(rows.len()))
        .map(|chunk| {
            let mut sum = 0.0;
            for &row in chunk {
                let datum = data[row];
                let base = row * ncols;
                let mut pred = 0.0;
                for col in 0..ncols {
                    pred += matrix[base + col] * model[col];
                }
                let residual = datum - pred;
                sum += 0.5 * residual * residual;
            }
            sum
        })
        .collect();
    let sum = partials.into_iter().sum::<f64>();
    sum + 0.5 * regularization * model.iter().map(|v| v * v).sum::<f64>()
}

fn adjoint_rows(matrix: &[f64], data: &[f64], rows: &[usize], ncols: usize) -> Vec<f64> {
    let partials: Vec<Vec<f64>> = rows
        .par_chunks(row_chunk_len(rows.len()))
        .map(|chunk| {
            let mut partial = vec![0.0; ncols];
            for &row in chunk {
                let datum = data[row];
                let base = row * ncols;
                for col in 0..ncols {
                    partial[col] += matrix[base + col] * datum;
                }
            }
            partial
        })
        .collect();
    reduce_partials(ncols, partials)
}

fn adjoint_residual_rows(
    matrix: &[f64],
    data: &[f64],
    model: &[f64],
    rows: &[usize],
    ncols: usize,
) -> Vec<f64> {
    let partials: Vec<Vec<f64>> = rows
        .par_chunks(row_chunk_len(rows.len()))
        .map(|chunk| {
            let mut partial = vec![0.0; ncols];
            for &row in chunk {
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
            }
            partial
        })
        .collect();
    reduce_partials(ncols, partials)
}

fn reduce_partials(ncols: usize, partials: Vec<Vec<f64>>) -> Vec<f64> {
    let mut out = vec![0.0; ncols];
    for partial in partials {
        add_vector_in_place(&mut out, &partial);
    }
    out
}

fn add_vector_in_place(out: &mut [f64], rhs: &[f64]) {
    for (value, increment) in out.iter_mut().zip(rhs) {
        *value += increment;
    }
}

fn row_chunk_len(row_count: usize) -> usize {
    let target_chunks = rayon::current_num_threads().max(1) * 4;
    row_count.div_ceil(target_chunks).max(1)
}
