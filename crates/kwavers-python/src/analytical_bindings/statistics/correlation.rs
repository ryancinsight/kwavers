//! Correlation and phase-sensitivity validation bindings.

use super::arrays::as_slices;
use kwavers_math::statistics;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Pearson correlation coefficient `r = cov(a,b)/(σ_a·σ_b) ∈ [−1, 1]` between a
/// simulation vector `a` and a reference vector `b` (book §19.2, Theorem). Returns
/// `0` for mismatched lengths or a constant input.
#[pyfunction]
#[pyo3(signature = (a, b))]
pub fn pearson(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let (a_s, b_s) = as_slices(&a, &b)?;
    Ok(statistics::pearson(a_s, b_s))
}

/// Same-frequency sinusoid Pearson curve `r(phi) = cos(phi)` for phase offsets
/// in radians (book Chapter 20 validation sensitivity).
#[pyfunction]
#[pyo3(signature = (phase_rad))]
pub fn phase_shift_correlation_curve(
    py: Python<'_>,
    phase_rad: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let phase_rad = phase_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = py
        .detach(|| statistics::phase_shift_correlation_curve(phase_rad))
        .map_err(PyValueError::new_err)?;
    Ok(result.to_pyarray(py).unbind())
}

/// Phase error in degrees for a target same-frequency sinusoid Pearson
/// correlation using the inverse theorem `phi = acos(r)`.
#[pyfunction]
#[pyo3(signature = (correlation))]
pub fn phase_error_degrees_for_correlation(correlation: f64) -> PyResult<f64> {
    statistics::phase_error_degrees_for_correlation(correlation).map_err(PyValueError::new_err)
}
