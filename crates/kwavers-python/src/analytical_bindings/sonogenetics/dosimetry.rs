//! Sonogenetics acoustic-dose bindings.

use kwavers_physics::analytical::sonogenetics;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute in-situ spatial-peak time-average intensity (ISPTA).
///
/// Args:
///     p_pa: Pressure time series [Pa].
///     dt_s: Sample interval [s].
///     rho: Density [kg/m³].
///     c: Sound speed [m/s].
///
/// Returns:
///     ISPTA [W/cm²].
#[pyfunction]
#[pyo3(signature = (p_pa, dt_s, rho, c))]
pub fn ispta_w_cm2(p_pa: PyReadonlyArray1<f64>, dt_s: f64, rho: f64, c: f64) -> PyResult<f64> {
    let p_s = p_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(sonogenetics::ispta_w_cm2(p_s, dt_s, rho, c))
}
