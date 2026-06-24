//! PyO3 bindings for `kwavers_physics::analytical::inverse` and
//! `kwavers_math::inverse_problems::parameter_selection`.

use kwavers_math::inverse_problems::parameter_selection;
use kwavers_physics::analytical::inverse as inverse_mod;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
    let flat = inverse_mod::helmholtz_1d_fd_matrix(n, k, dx);
    let arr2d =
        Array2::from_shape_vec((n, n), flat).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
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
    let m = matrix.as_array();
    let flat: Vec<f64> = (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| m[[i, j]]))
        .collect();
    let result = inverse_mod::matrix_singular_values(&flat, nrows, ncols);
    Ok(result.into_pyarray(py).unbind())
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
    let a_arr = a.as_array();
    let a_flat: Vec<f64> = (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| a_arr[[i, j]]))
        .collect();
    let b_s = b
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lam_s = lambdas
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (res, sol) = inverse_mod::tikhonov_lcurve(&a_flat, b_s, nrows, ncols, lam_s);
    Ok((res.into_pyarray(py).unbind(), sol.into_pyarray(py).unbind()))
}

/// Solve a Born-inversion problem with Tikhonov regularisation.
///
/// Args:
///     g_real: Real part of the Green's function matrix (nrows × ncols).
///     g_imag: Imaginary part (nrows × ncols).
///     y_real: Real part of measurement vector (length nrows).
///     y_imag: Imaginary part (length nrows).
///     nrows: Number of rows.
///     ncols: Number of columns.
///     lambda: Regularisation parameter.
///
/// Returns:
///     (real_solution, imag_solution) tuple.
#[pyfunction]
#[pyo3(signature = (g_real, g_imag, y_real, y_imag, nrows, ncols, lambda))]
pub fn born_inversion_regularized(
    py: Python<'_>,
    g_real: PyReadonlyArray2<f64>,
    g_imag: PyReadonlyArray2<f64>,
    y_real: PyReadonlyArray1<f64>,
    y_imag: PyReadonlyArray1<f64>,
    nrows: usize,
    ncols: usize,
    lambda: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let gr = g_real.as_array();
    let gi = g_imag.as_array();
    let gr_flat: Vec<f64> = (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| gr[[i, j]]))
        .collect();
    let gi_flat: Vec<f64> = (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| gi[[i, j]]))
        .collect();
    let yr_s = y_real
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let yi_s = y_imag
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (re, im) = inverse_mod::born_inversion_regularized(
        &gr_flat, &gi_flat, yr_s, yi_s, nrows, ncols, lambda,
    );
    Ok((re.into_pyarray(py).unbind(), im.into_pyarray(py).unbind()))
}

/// Compute the adjoint-gradient convergence curve.
///
/// Args:
///     n_iter: Number of iterations.
///     initial_error: Initial relative error.
///     decay: Per-iteration error decay factor.
///
/// Returns:
///     Error array of length *n_iter*.
#[pyfunction]
#[pyo3(signature = (n_iter, initial_error, decay))]
pub fn adjoint_gradient_convergence(
    py: Python<'_>,
    n_iter: usize,
    initial_error: f64,
    decay: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = inverse_mod::adjoint_gradient_convergence(n_iter, initial_error, decay);
    Ok(result.into_pyarray(py).unbind())
}

/// L-curve corner index (point of maximum curvature of the log–log
/// residual-vs-model-norm trade-off, Hansen 1992; book §18.7). Returns the
/// interior index of the maximum-curvature sample, or `None` if the inputs are
/// too short (`n < 5`), mismatched, non-positive, or `lambdas` not increasing.
///
/// Args:
///     residual_norms: ‖F(m_λ)−d‖ over the λ sweep.
///     model_norms: ‖m_λ‖ over the λ sweep.
///     lambdas: Regularization weights (strictly increasing, > 0).
///
/// Returns:
///     Corner index (int) or None.
#[pyfunction]
#[pyo3(signature = (residual_norms, model_norms, lambdas))]
pub fn l_curve_corner(
    residual_norms: PyReadonlyArray1<f64>,
    model_norms: PyReadonlyArray1<f64>,
    lambdas: PyReadonlyArray1<f64>,
) -> PyResult<Option<usize>> {
    let r = residual_norms
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let m = model_norms
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let l = lambdas
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(parameter_selection::l_curve_corner(r, m, l))
}

/// Morozov discrepancy-principle regularization weight `λ*`: the `λ` at which the
/// data-residual norm equals `τ·δ` (noise level `δ`, safety factor `τ ≥ 1`),
/// found by linear interpolation (book §18.7). Returns `None` if `τ·δ` is outside
/// the sampled residual range or the inputs are invalid.
///
/// Args:
///     lambdas: Regularization weights over the sweep.
///     residual_norms: ‖F(m_λ)−d‖ over the sweep (monotone non-decreasing).
///     noise_level: Noise level δ.
///     tau: Safety factor τ ≥ 1.
///
/// Returns:
///     λ* (float) or None.
#[pyfunction]
#[pyo3(signature = (lambdas, residual_norms, noise_level, tau))]
pub fn morozov_lambda(
    lambdas: PyReadonlyArray1<f64>,
    residual_norms: PyReadonlyArray1<f64>,
    noise_level: f64,
    tau: f64,
) -> PyResult<Option<f64>> {
    let l = lambdas
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let r = residual_norms
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(parameter_selection::morozov_lambda(l, r, noise_level, tau))
}
