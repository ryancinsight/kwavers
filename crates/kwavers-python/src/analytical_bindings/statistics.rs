//! PyO3 bindings for `kwavers_math::statistics` validation metrics (book §19).

use kwavers_math::statistics;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn as_slices<'a>(
    a: &'a PyReadonlyArray1<f64>,
    b: &'a PyReadonlyArray1<f64>,
) -> PyResult<(&'a [f64], &'a [f64])> {
    let a_s = a
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let b_s = b
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((a_s, b_s))
}

/// Pearson correlation coefficient `r = cov(a,b)/(σ_a·σ_b) ∈ [−1, 1]` between a
/// simulation vector `a` and a reference vector `b` (book §19.2, Theorem). Returns
/// `0` for mismatched lengths or a constant input.
#[pyfunction]
#[pyo3(signature = (a, b))]
pub fn pearson(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let (a_s, b_s) = as_slices(&a, &b)?;
    Ok(statistics::pearson(a_s, b_s))
}

/// Root-mean-square error `RMSE = √(mean((a−b)²))` between `a` and `b` (book §19.3).
#[pyfunction]
#[pyo3(signature = (a, b))]
pub fn rmse(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let (a_s, b_s) = as_slices(&a, &b)?;
    Ok(statistics::rmse(a_s, b_s))
}

/// Peak signal-to-noise ratio `PSNR = 20·log₁₀(MAX_b / RMSE(a,b))` [dB], with
/// `MAX_b` the peak of the reference `b` (book §19.3, Theorem). Returns `+∞` when
/// `a == b` (zero error).
#[pyfunction]
#[pyo3(signature = (a, b))]
pub fn psnr(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let (a_s, b_s) = as_slices(&a, &b)?;
    Ok(statistics::psnr(a_s, b_s))
}
