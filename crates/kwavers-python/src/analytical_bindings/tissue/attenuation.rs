//! Tissue attenuation and dispersion analytical bindings.

use kwavers_physics::analytical::tissue;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute frequency-dependent tissue absorption in dB/cm.
///
/// Args:
///     f_mhz: Frequency array [MHz].
///     tissue: Tissue name string.
///
/// Returns:
///     Absorption array [dB/cm].
#[pyfunction]
#[pyo3(signature = (f_mhz, tissue))]
pub fn tissue_absorption_db_cm(
    py: Python<'_>,
    f_mhz: PyReadonlyArray1<f64>,
    tissue: String,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::tissue_absorption_db_cm(f_s, &tissue);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Kramers–Kronig consistent sound speed dispersion.
///
/// Args:
///     f_hz: Frequency array [Hz].
///     alpha0: Attenuation coefficient [Np/m/Hz^y].
///     y: Power-law exponent.
///     f_ref_hz: Reference frequency [Hz].
///     c_ref: Sound speed at *f_ref_hz* [m/s].
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (f_hz, alpha0, y, f_ref_hz, c_ref))]
pub fn kramers_kronig_sound_speed(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    alpha0: f64,
    y: f64,
    f_ref_hz: f64,
    c_ref: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::kramers_kronig_sound_speed(f_s, alpha0, y, f_ref_hz, c_ref);
    Ok(result.into_pyarray(py).unbind())
}
