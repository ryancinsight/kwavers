//! Water-property analytical bindings.

use kwavers_physics::analytical::tissue;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the sound speed of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
pub fn water_sound_speed_temperature(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::water_sound_speed_temperature(t_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the density of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Density array [kg/m³].
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
pub fn water_density_temperature(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::water_density_temperature(t_s);
    Ok(result.into_pyarray(py).unbind())
}
