//! Inverse reconstruction fixture and Born-inversion bindings.

use super::arrays::{array2_from_flat, flatten_array2};
use kwavers_physics::analytical::inverse as inverse_mod;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Deterministic Gaussian-deconvolution fixture for the Chapter 17 L-curve.
#[pyfunction]
#[pyo3(signature = (n, sigma, perturbation_scale=0.01))]
pub fn gaussian_deconvolution_fixture(
    py: Python<'_>,
    n: usize,
    sigma: f64,
    perturbation_scale: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let fixture = inverse_mod::gaussian_deconvolution_fixture(n, sigma, perturbation_scale)
        .map_err(PyValueError::new_err)?;
    Ok((
        array2_from_flat(py, n, n, fixture.matrix)?,
        fixture.truth_signal.into_pyarray(py).unbind(),
        fixture.observed_signal.into_pyarray(py).unbind(),
    ))
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
    let gr_flat = flatten_array2(g_real.as_array(), nrows, ncols);
    let gi_flat = flatten_array2(g_imag.as_array(), nrows, ncols);
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
