//! Thermal-index and CEM43 safety bindings.

use kwavers_physics::analytical::safety;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compute the Thermal Index for soft tissue (TIS).
///
/// Args:
///     wstp_mw: W_STP — time-averaged power at the surface [mW].
///     f_mhz: Frequency [MHz].
///
/// Returns:
///     TIS value.
#[pyfunction]
#[pyo3(signature = (wstp_mw, f_mhz))]
pub fn thermal_index_soft_tissue(wstp_mw: f64, f_mhz: f64) -> PyResult<f64> {
    Ok(safety::thermal_index_soft_tissue(wstp_mw, f_mhz))
}

/// Compute the Thermal Index for bone (TIB).
///
/// Args:
///     w_mw: Beam power at bone surface [mW].
///     f_mhz: Frequency [MHz].
///
/// Returns:
///     TIB value.
#[pyfunction]
#[pyo3(signature = (w_mw, f_mhz))]
pub fn thermal_index_bone(w_mw: f64, f_mhz: f64) -> PyResult<f64> {
    Ok(safety::thermal_index_bone(w_mw, f_mhz))
}

/// Compute the Thermal Index for cranial bone (TIC).
///
/// Frequency-independent (IEC 62359 §8.5): TIC = W_0 / (40·D_eq).
///
/// Args:
///     w0_mw: Total acoustic power at the transducer face [mW].
///     aperture_diameter_cm: Equivalent aperture diameter D_eq = sqrt(4·A/pi) [cm].
///
/// Returns:
///     TIC value.
#[pyfunction]
#[pyo3(signature = (w0_mw, aperture_diameter_cm))]
pub fn thermal_index_cranial(w0_mw: f64, aperture_diameter_cm: f64) -> PyResult<f64> {
    Ok(safety::thermal_index_cranial(w0_mw, aperture_diameter_cm))
}

/// Compute the cumulative CEM43 thermal dose over a temperature time series.
///
/// Args:
///     t_celsius: Temperature time series [°C].
///     dt_s: Time-step [s].
///
/// Returns:
///     Cumulative CEM43 array [min].
#[pyfunction]
#[pyo3(signature = (t_celsius, dt_s))]
pub fn cem43_cumulative(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
    dt_s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let temperatures = t_s.to_vec();
    let result = py
        .detach(move || crate::response::thermal::cem43_cumulative(&temperatures, dt_s))
        .map_err(PyValueError::new_err)?;
    Ok(result.to_pyarray(py).unbind())
}

/// Chapter 7 closed-loop focal-temperature and CEM43 dose fixture.
#[pyfunction]
#[pyo3(signature = (n_steps=60, dt_s=0.5, body_temperature_c=37.0, target_temperature_c=60.0, seed=42))]
pub fn closed_loop_cem43_fixture<'py>(
    py: Python<'py>,
    n_steps: usize,
    dt_s: f64,
    body_temperature_c: f64,
    target_temperature_c: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let fixture = safety::closed_loop_cem43_fixture(
        n_steps,
        dt_s,
        body_temperature_c,
        target_temperature_c,
        seed,
    )
    .map_err(|source| PyValueError::new_err(source.to_string()))?;

    let out = PyDict::new(py);
    out.set_item("time_s", fixture.time_s.to_pyarray(py))?;
    out.set_item(
        "fixed_temperature_c",
        fixture.fixed_temperature_c.to_pyarray(py),
    )?;
    out.set_item(
        "feedback_temperature_c",
        fixture.feedback_temperature_c.to_pyarray(py),
    )?;
    out.set_item(
        "underdrive_temperature_c",
        fixture.underdrive_temperature_c.to_pyarray(py),
    )?;
    out.set_item("fixed_cem43_min", fixture.fixed_cem43_min.to_pyarray(py))?;
    out.set_item(
        "feedback_cem43_min",
        fixture.feedback_cem43_min.to_pyarray(py),
    )?;
    out.set_item(
        "underdrive_cem43_min",
        fixture.underdrive_cem43_min.to_pyarray(py),
    )?;
    Ok(out)
}
