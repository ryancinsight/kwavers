//! Forward-operator and regularization-curve bindings.

use super::arrays::{array2_from_flat, flatten_array2};
use kwavers_physics::analytical::inverse as inverse_mod;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Build the 1-D Helmholtz finite-difference matrix (Dirichlet BCs).
///
/// Returns an (n × n) sparse-dense matrix A such that (A + k²I)u = f.
///
/// Args:
///     n: Grid points.
///     k: Wave number [rad/m].
///     dx: Grid spacing [m].
///
/// Returns:
///     Dense ndarray of shape (n, n).
#[pyfunction]
#[pyo3(signature = (n, k, dx))]
pub fn helmholtz_1d_fd_matrix(
    py: Python<'_>,
    n: usize,
    k: f64,
    dx: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    array2_from_flat(py, n, n, inverse_mod::helmholtz_1d_fd_matrix(n, k, dx))
}

/// Compute the singular values of a dense matrix.
///
/// Args:
///     matrix: Dense matrix (nrows × ncols).
///     nrows: Number of rows.
///     ncols: Number of columns.
///
/// Returns:
///     Singular values in descending order.
#[pyfunction]
#[pyo3(signature = (matrix, nrows, ncols))]
pub fn matrix_singular_values(
    py: Python<'_>,
    matrix: PyReadonlyArray2<f64>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let flat = flatten_array2(&matrix)?;
    let result = inverse_mod::matrix_singular_values(&flat, nrows, ncols);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the L-curve (residual norm vs. solution norm) for Tikhonov regularisation.
///
/// Args:
///     a: System matrix (nrows × ncols).
///     b: Right-hand-side vector (length nrows).
///     nrows: Number of rows.
///     ncols: Number of columns.
///     lambdas: Regularisation parameter array.
///
/// Returns:
///     (residual_norms, solution_norms) tuple.
#[pyfunction]
#[pyo3(signature = (a, b, nrows, ncols, lambdas))]
pub fn tikhonov_lcurve(
    py: Python<'_>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray1<f64>,
    nrows: usize,
    ncols: usize,
    lambdas: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let a_flat = flatten_array2(&a)?;
    let b_s = b
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lam_s = lambdas
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (res, sol) = inverse_mod::tikhonov_lcurve(&a_flat, b_s, nrows, ncols, lam_s);
    Ok((res.to_pyarray(py).unbind(), sol.to_pyarray(py).unbind()))
}
