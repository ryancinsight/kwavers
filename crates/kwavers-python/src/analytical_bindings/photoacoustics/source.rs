//! Source and pressure-signal photoacoustic bindings.

use kwavers_physics::analytical::photoacoustics;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compute the photoacoustic pressure signal from an absorbing sphere.
///
/// Args:
///     t_arr: Time array [s].
///     r0_m: Sphere radius [m].
///     gamma: Grüneisen parameter.
///     mua_per_m: Absorption coefficient [1/m].
///     c: Sound speed [m/s].
///     r_det_m: Detector distance [m].
///     initial_pressure_pa: Initial pressure rise [Pa].
///
/// Returns:
///     Pressure signal array [Pa].
#[pyfunction]
#[pyo3(signature = (t_arr, r0_m, gamma, mua_per_m, c, r_det_m, initial_pressure_pa))]
pub fn pa_sphere_pressure_signal(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    r0_m: f64,
    gamma: f64,
    mua_per_m: f64,
    c: f64,
    r_det_m: f64,
    initial_pressure_pa: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::pa_sphere_pressure_signal(
        t_s,
        r0_m,
        gamma,
        mua_per_m,
        c,
        r_det_m,
        initial_pressure_pa,
    );
    Ok(result.into_pyarray(py).unbind())
}

/// Compute a Gaussian absorber initial-pressure profile and bipolar PA signal.
#[pyfunction]
#[pyo3(signature = (
    depth_axis_m,
    time_axis_s,
    gruneisen,
    absorption_per_m,
    fluence_j_m2,
    center_m,
    sigma_m,
    sound_speed_m_s
))]
pub fn gaussian_absorber_photoacoustic_profile<'py>(
    py: Python<'py>,
    depth_axis_m: PyReadonlyArray1<f64>,
    time_axis_s: PyReadonlyArray1<f64>,
    gruneisen: f64,
    absorption_per_m: f64,
    fluence_j_m2: f64,
    center_m: f64,
    sigma_m: f64,
    sound_speed_m_s: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let depth_axis = depth_axis_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let time_axis = time_axis_s
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (initial_pressure, surface_signal) =
        photoacoustics::gaussian_absorber_photoacoustic_profile(
            depth_axis,
            time_axis,
            gruneisen,
            absorption_per_m,
            fluence_j_m2,
            center_m,
            sigma_m,
            sound_speed_m_s,
        )
        .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("initial_pressure_pa", initial_pressure.into_pyarray(py))?;
    out.set_item("surface_signal_pa_per_m", surface_signal.into_pyarray(py))?;
    Ok(out)
}
