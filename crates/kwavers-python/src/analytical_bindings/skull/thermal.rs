//! Skull surface-heating bindings.

use kwavers_physics::analytical::skull as skull_mod;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute skull surface temperature rise due to a heat-flux boundary.
///
/// Args:
///     t_arr: Time array `s`.
///     heat_flux: Applied heat flux [W/m²].
///     k_skull: Skull thermal conductivity [W/(m·K)].
///     rho_skull: Skull density [kg/m³].
///     cp_skull: Skull specific heat capacity [J/(kg·K)].
///
/// Returns:
///     Surface temperature-rise array [°C].
#[pyfunction]
#[pyo3(signature = (t_arr, heat_flux, k_skull, rho_skull, cp_skull))]
pub fn skull_surface_temperature_rise(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    heat_flux: f64,
    k_skull: f64,
    rho_skull: f64,
    cp_skull: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        skull_mod::skull_surface_temperature_rise(t_s, heat_flux, k_skull, rho_skull, cp_skull);
    Ok(result.to_pyarray(py).unbind())
}
