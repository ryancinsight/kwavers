//! Error-metric validation bindings.

use super::arrays::as_slices;
use kwavers_math::statistics;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// PSNR in dB for relative RMSE values using `PSNR = -20 log10(relative_rmse)`.
#[pyfunction]
#[pyo3(signature = (relative_rmse))]
pub fn validation_psnr_from_relative_rmse(
    py: Python<'_>,
    relative_rmse: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let relative_rmse = relative_rmse
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = py
        .detach(|| statistics::validation_psnr_from_relative_rmse(relative_rmse))
        .map_err(PyValueError::new_err)?;
    Ok(result.to_pyarray(py).unbind())
}

/// Root-mean-square error `RMSE = √(mean((a−b)²))` between `a` and `b` (book §19.3).
#[pyfunction]
#[pyo3(signature = (a, b))]
pub fn rmse(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let (a_s, b_s) = as_slices(&a, &b)?;
    Ok(statistics::rmse(a_s, b_s))
}

/// Peak signal-to-noise ratio `PSNR = 20·log₁₀(MAX_b / RMSE(a,b))` `dB`, with
/// `MAX_b` the peak of the reference `b` (book §19.3, Theorem). Returns `+∞` when
/// `a == b` (zero error).
#[pyfunction]
#[pyo3(signature = (a, b))]
pub fn psnr(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let (a_s, b_s) = as_slices(&a, &b)?;
    Ok(statistics::psnr(a_s, b_s))
}
