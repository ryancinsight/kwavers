//! PyO3 bindings: basic wave propagation (standing, plane, spherical,
//! reflection/transmission, power-law attenuation, Stokes–Kirchhoff absorption).

use kwavers_physics::analytical::wave;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Compute a 1-D standing-wave pressure field.
///
/// p(x, t) = p0 * sin(k*x) * cos(omega*t)
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

/// Compute pressure and particle velocity for a 1-D progressive plane wave.
#[pyfunction]
#[pyo3(signature = (amplitude_pa, k, x, omega_t, density_kg_m3, sound_speed_m_s))]
pub fn plane_wave_pressure_velocity_1d(
    py: Python<'_>,
    amplitude_pa: f64,
    k: f64,
    x: PyReadonlyArray1<f64>,
    omega_t: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let x_slice = x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (pressure, velocity) = wave::plane_wave_pressure_velocity_1d(
        amplitude_pa,
        k,
        x_slice,
        omega_t,
        density_kg_m3,
        sound_speed_m_s,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        pressure.into_pyarray(py).unbind(),
        velocity.into_pyarray(py).unbind(),
    ))
}

/// Gaussian-modulated cosine pulse over a 1-D coordinate axis.
#[pyfunction]
#[pyo3(signature = (x, center_m, sigma_m, wavelength_m, amplitude_pa))]
pub fn gaussian_modulated_pulse_1d(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    center_m: f64,
    sigma_m: f64,
    wavelength_m: f64,
    amplitude_pa: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        wave::gaussian_modulated_pulse_1d(x_slice, center_m, sigma_m, wavelength_m, amplitude_pa)
            .map_err(PyValueError::new_err)?;
    Ok(result.into_pyarray(py).unbind())
}

/// d'Alembert zero-initial-velocity split-pulse solution on a uniform 1-D axis.
#[pyfunction]
#[pyo3(signature = (x, initial_pressure, shift_m))]
pub fn dalembert_split_solution_1d(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    initial_pressure: PyReadonlyArray1<f64>,
    shift_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let p_slice = initial_pressure
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::dalembert_split_solution_1d(x_slice, p_slice, shift_m)
        .map_err(PyValueError::new_err)?;
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

/// Compute normalized spherical and cylindrical spreading intensity envelopes.
#[pyfunction]
#[pyo3(signature = (r,))]
pub fn geometric_spreading_intensity_envelopes(
    py: Python<'_>,
    r: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let r_slice = r
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (spherical, cylindrical) =
        wave::geometric_spreading_intensity_envelopes(r_slice).map_err(PyValueError::new_err)?;
    Ok((
        spherical.into_pyarray(py).unbind(),
        cylindrical.into_pyarray(py).unbind(),
    ))
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
