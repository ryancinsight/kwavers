//! PyO3 bindings: basic wave propagation (standing, plane, spherical,
//! reflection/transmission, power-law attenuation, Stokes–Kirchhoff absorption).

use kwavers_physics::analytical::wave;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute a 1-D standing-wave pressure field.
///
/// p(x, t) = 2 * p0 * cos(k*x) * cos(omega*t)
///
/// Args:
///     p0: Peak amplitude [Pa].
///     k: Wave number [rad/m].
///     x: Spatial positions [m] (1-D array).
///     omega_t: Phase angle omega*t [rad].
///
/// Returns:
///     Pressure array [Pa] of the same length as *x*.
#[pyfunction]
#[pyo3(signature = (p0, k, x, omega_t))]
pub fn standing_wave_1d(
    py: Python<'_>,
    p0: f64,
    k: f64,
    x: PyReadonlyArray1<f64>,
    omega_t: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::standing_wave_1d(p0, k, x_slice, omega_t);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute a 1-D plane-wave pressure field.
///
/// p(x, t) = amplitude * cos(k*x - omega*t)
///
/// Args:
///     amplitude: Amplitude [Pa].
///     k: Wave number [rad/m].
///     x: Spatial positions [m].
///     omega_t: Phase angle omega*t [rad].
///
/// Returns:
///     Pressure array [Pa].
#[pyfunction]
#[pyo3(signature = (amplitude, k, x, omega_t))]
pub fn plane_wave_pressure_1d(
    py: Python<'_>,
    amplitude: f64,
    k: f64,
    x: PyReadonlyArray1<f64>,
    omega_t: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::plane_wave_pressure_1d(amplitude, k, x_slice, omega_t);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute spherical-wave pressure at radial distances *r*.
///
/// p(r) = amplitude / r * cos(k*r)
///
/// Args:
///     amplitude: Source strength [Pa·m].
///     k: Wave number [rad/m].
///     r: Radial distances [m].
///
/// Returns:
///     Pressure array [Pa].
#[pyfunction]
#[pyo3(signature = (amplitude, k, r))]
pub fn spherical_wave_pressure(
    py: Python<'_>,
    amplitude: f64,
    k: f64,
    r: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let r_slice = r
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::spherical_wave_pressure(amplitude, k, r_slice);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the pressure reflection coefficient at a planar interface.
///
/// R = (z2 - z1) / (z2 + z1)
///
/// Args:
///     z1: Acoustic impedance of medium 1 [Pa·s/m].
///     z2: Acoustic impedance of medium 2 [Pa·s/m].
///
/// Returns:
///     Pressure reflection coefficient (dimensionless).
#[pyfunction]
#[pyo3(signature = (z1, z2))]
pub fn reflection_pressure_coeff(z1: f64, z2: f64) -> PyResult<f64> {
    Ok(wave::reflection_pressure_coeff(z1, z2))
}

/// Compute the pressure transmission coefficient at a planar interface.
///
/// T = 2*z2 / (z2 + z1)
///
/// Args:
///     z1: Acoustic impedance of medium 1 [Pa·s/m].
///     z2: Acoustic impedance of medium 2 [Pa·s/m].
///
/// Returns:
///     Pressure transmission coefficient (dimensionless).
#[pyfunction]
#[pyo3(signature = (z1, z2))]
pub fn transmission_pressure_coeff(z1: f64, z2: f64) -> PyResult<f64> {
    Ok(wave::transmission_pressure_coeff(z1, z2))
}

/// Compute power-law attenuation α(f) = α0 * f^y in Np/m.
///
/// Args:
///     f_hz: Frequency array [Hz].
///     alpha0: Attenuation coefficient at 1 Hz [Np/m/Hz^y].
///     y: Frequency power-law exponent.
///
/// Returns:
///     Attenuation array [Np/m].
#[pyfunction]
#[pyo3(signature = (f_hz, alpha0, y))]
pub fn power_law_attenuation_np_m(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    alpha0: f64,
    y: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_slice = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::power_law_attenuation_np_m(f_slice, alpha0, y);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute power-law absorption α(f) = α0 * f^y in dB/(cm·MHz^y).
///
/// Args:
///     f_mhz: Frequency array [MHz].
///     alpha0: Coefficient [dB/(cm·MHz^y)].
///     y: Power-law exponent.
///
/// Returns:
///     Absorption array [dB/cm].
#[pyfunction]
#[pyo3(signature = (f_mhz, alpha0, y))]
pub fn absorption_power_law_db_cm(
    py: Python<'_>,
    f_mhz: PyReadonlyArray1<f64>,
    alpha0: f64,
    y: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_slice = f_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::absorption_power_law_db_cm(f_slice, alpha0, y);
    Ok(result.into_pyarray(py).unbind())
}

/// Stokes-Kirchhoff thermoviscous absorption coefficient [Np/m].
///
/// Classical result (Stokes 1845, Kirchhoff 1868):
///   α_SK(ω) = δ · ω² / (2 · c₀³)   [Np/m],   ω = 2πf
///
/// Args:
///     freqs_hz: Frequency array [Hz].
///     delta_m2_s: Acoustic diffusivity δ [m²/s].
///     c0: Small-signal sound speed [m/s].
///
/// Returns:
///     Absorption coefficient array [Np/m].
///
/// Reference:
///     Pierce (1989) Acoustics, §10.1, Eq. 10.1.11.
#[pyfunction]
#[pyo3(signature = (freqs_hz, delta_m2_s, c0))]
pub fn stokes_kirchhoff_absorption_np_m(
    py: Python<'_>,
    freqs_hz: PyReadonlyArray1<f64>,
    delta_m2_s: f64,
    c0: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_slice = freqs_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::stokes_kirchhoff_absorption_np_m(f_slice, delta_m2_s, c0);
    Ok(result.into_pyarray(py).unbind())
}
