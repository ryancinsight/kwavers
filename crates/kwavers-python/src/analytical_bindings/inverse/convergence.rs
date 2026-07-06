//! Convergence-curve bindings for inverse-problem examples.

use kwavers_physics::analytical::inverse as inverse_mod;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

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
    Ok(result.to_pyarray(py).unbind())
}

/// Exponential convergence curve with additive floor.
#[pyfunction]
#[pyo3(signature = (epochs, initial_value, time_constant, floor))]
pub fn exponential_convergence_curve(
    py: Python<'_>,
    epochs: PyReadonlyArray1<f64>,
    initial_value: f64,
    time_constant: f64,
    floor: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let epochs = epochs
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        inverse_mod::exponential_convergence_curve(epochs, initial_value, time_constant, floor)
            .map_err(PyValueError::new_err)?;
    Ok(result.to_pyarray(py).unbind())
}

