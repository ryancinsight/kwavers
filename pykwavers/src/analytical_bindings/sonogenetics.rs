//! PyO3 bindings for `kwavers::physics::analytical::sonogenetics`.

use kwavers::physics::analytical::sonogenetics;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the 1-D acoustic radiation force density.
///
/// F = 2 * alpha * I / c
///
/// Args:
///     intensity_w_m2: Intensity array [W/m²].
///     alpha_np_m: Attenuation [Np/m].
///     c: Sound speed [m/s].
///
/// Returns:
///     Radiation force density array [N/m³].
#[pyfunction]
#[pyo3(signature = (intensity_w_m2, alpha_np_m, c))]
pub fn radiation_force_1d(
    py: Python<'_>,
    intensity_w_m2: PyReadonlyArray1<f64>,
    alpha_np_m: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let i_s = intensity_w_m2
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = sonogenetics::radiation_force_1d(i_s, alpha_np_m, c);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the steady acoustic streaming velocity (Eckart streaming).
///
/// Args:
///     i_w_m2: Beam intensity [W/m²].
///     mu_pa_s: Dynamic viscosity [Pa·s].
///     alpha_np_m: Attenuation [Np/m].
///     c: Sound speed [m/s].
///     l_m: Beam propagation length [m].
///
/// Returns:
///     Streaming velocity [m/s].
#[pyfunction]
#[pyo3(signature = (i_w_m2, mu_pa_s, alpha_np_m, c, l_m))]
pub fn acoustic_streaming_velocity(
    i_w_m2: f64,
    mu_pa_s: f64,
    alpha_np_m: f64,
    c: f64,
    l_m: f64,
) -> PyResult<f64> {
    Ok(sonogenetics::acoustic_streaming_velocity(
        i_w_m2, mu_pa_s, alpha_np_m, c, l_m,
    ))
}

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
pub fn ispta_w_cm2(
    p_pa: PyReadonlyArray1<f64>,
    dt_s: f64,
    rho: f64,
    c: f64,
) -> PyResult<f64> {
    let p_s = p_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(sonogenetics::ispta_w_cm2(p_s, dt_s, rho, c))
}
