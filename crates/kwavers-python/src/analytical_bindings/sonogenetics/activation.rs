//! Mechanosensitive-channel activation bindings.

use kwavers_physics::analytical::sonogenetics;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the Hill activation probability for mechanosensitive channels.
///
/// P(p) = p^n / (p_threshold^n + p^n)
///
/// Args:
///     pressure_arr: Pressure amplitude array [Pa].
///     p_threshold_pa: Half-activation pressure [Pa].
///     hill_n: Hill coefficient.
///
/// Returns:
///     Activation probability array (0–1).
#[pyfunction]
#[pyo3(signature = (pressure_arr, p_threshold_pa, hill_n))]
pub fn hill_activation_probability(
    py: Python<'_>,
    pressure_arr: PyReadonlyArray1<f64>,
    p_threshold_pa: f64,
    hill_n: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = pressure_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = sonogenetics::hill_activation_probability(p_s, p_threshold_pa, hill_n);
    Ok(result.to_pyarray(py).unbind())
}

