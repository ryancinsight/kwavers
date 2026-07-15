//! Layered skull transmission bindings.

use eunomia::Complex64;
use kwavers_physics::analytical::skull as skull_mod;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the complex transmission coefficient through a skull layer
/// using the transfer-matrix method.
///
/// Args:
///     f_hz: Frequency array [Hz].
///     z_water: Water impedance [Pa·s/m].
///     z_skull: Skull impedance [Pa·s/m].
///     z_brain: Brain impedance [Pa·s/m].
///     c_skull: Skull sound speed [m/s].
///     d_skull_m: Skull thickness [m].
///
/// Returns:
///     Python complex number T = |T| * exp(i*phi).
#[pyfunction]
#[pyo3(signature = (f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m))]
pub fn skull_transfer_matrix_transmission(
    py: Python<'_>,
    f_hz: f64,
    z_water: f64,
    z_skull: f64,
    z_brain: f64,
    c_skull: f64,
    d_skull_m: f64,
) -> PyResult<Py<PyAny>> {
    let c: Complex64 = skull_mod::skull_transfer_matrix_transmission(
        f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m,
    );
    let builtins = py.import("builtins")?;
    let py_complex = builtins.getattr("complex")?;
    Ok(py_complex.call1((c.re, c.im))?.into())
}

/// Compute the skull transmission spectrum (magnitude and phase).
///
/// Args:
///     f_hz: Frequency array [Hz].
///     z_water: Water impedance [Pa·s/m].
///     z_skull: Skull impedance [Pa·s/m].
///     z_brain: Brain impedance [Pa·s/m].
///     c_skull: Skull sound speed [m/s].
///     d_skull_m: Skull thickness [m].
///
/// Returns:
///     (magnitude_array, phase_array_rad) tuple.
#[pyfunction]
#[pyo3(signature = (f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m))]
pub fn skull_transmission_spectrum(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    z_water: f64,
    z_skull: f64,
    z_brain: f64,
    c_skull: f64,
    d_skull_m: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let f_s = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (mag, phase) =
        skull_mod::skull_transmission_spectrum(f_s, z_water, z_skull, z_brain, c_skull, d_skull_m);
    Ok((mag.to_pyarray(py).unbind(), phase.to_pyarray(py).unbind()))
}
