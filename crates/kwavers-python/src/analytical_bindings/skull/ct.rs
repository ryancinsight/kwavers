//! Hounsfield-unit skull material conversion bindings.

use kwavers_physics::analytical::skull as skull_mod;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Convert Hounsfield units to sound speed using the Schneider model.
///
/// Args:
///     hu: HU array.
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (hu,))]
pub fn hu_to_sound_speed_schneider(
    py: Python<'_>,
    hu: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let h_s = hu
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::hu_to_sound_speed_schneider(h_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Convert Hounsfield units to density using the Schneider model.
///
/// Args:
///     hu: HU array.
///
/// Returns:
///     Density array [kg/m³].
#[pyfunction]
#[pyo3(signature = (hu,))]
pub fn hu_to_density_schneider(
    py: Python<'_>,
    hu: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let h_s = hu
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::hu_to_density_schneider(h_s);
    Ok(result.to_pyarray(py).unbind())
}

