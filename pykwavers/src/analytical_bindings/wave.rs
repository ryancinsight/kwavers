//! PyO3 bindings for `kwavers::physics::analytical::wave`.

use kwavers::physics::analytical::wave;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
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

/// Compute FDTD numerical phase error for a 1-D Yee grid.
///
/// Args:
///     kh: k*h (wave-number times grid spacing) array.
///     cfl: Courant–Friedrichs–Lewy number.
///
/// Returns:
///     Relative phase error array.
#[pyfunction]
#[pyo3(signature = (kh, cfl))]
pub fn fdtd_phase_error_1d(
    py: Python<'_>,
    kh: PyReadonlyArray1<f64>,
    cfl: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::fdtd_phase_error_1d(kh_slice, cfl);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute PSTD numerical phase error.
///
/// Args:
///     kh: k*h array.
///
/// Returns:
///     Relative phase error array.
#[pyfunction]
#[pyo3(signature = (kh,))]
pub fn pstd_phase_error(py: Python<'_>, kh: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::pstd_phase_error(kh_slice);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute k-space dispersion-correction phase error.
///
/// Args:
///     kh: k*h array.
///     cfl: CFL number.
///
/// Returns:
///     Residual phase error array after k-space correction.
#[pyfunction]
#[pyo3(signature = (kh, cfl))]
pub fn kspace_correction_error(
    py: Python<'_>,
    kh: PyReadonlyArray1<f64>,
    cfl: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::kspace_correction_error(kh_slice, cfl);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the FDTD CFL stability limit for an n-dimensional grid.
///
/// CFL_max = 1 / sqrt(ndim)
///
/// Args:
///     ndim: Number of spatial dimensions (1, 2, or 3).
///
/// Returns:
///     Maximum stable CFL number.
#[pyfunction]
#[pyo3(signature = (ndim,))]
pub fn fdtd_cfl_limit(ndim: u32) -> PyResult<f64> {
    Ok(wave::fdtd_cfl_limit(ndim))
}

/// Compute the n-th Fubini harmonic amplitude for a finite-amplitude wave.
///
/// Args:
///     n: Harmonic index (1-based).
///     sigma: Fubini variable (normalised propagation distance).
///
/// Returns:
///     Normalised harmonic amplitude.
#[pyfunction]
#[pyo3(signature = (n, sigma))]
pub fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> PyResult<f64> {
    Ok(wave::fubini_harmonic_amplitude(n, sigma))
}

/// Compute the full Fubini harmonic spectrum up to order *n_max*.
///
/// Args:
///     n_max: Highest harmonic order to include.
///     sigma: Fubini variable.
///
/// Returns:
///     Array of normalised harmonic amplitudes, length *n_max*.
#[pyfunction]
#[pyo3(signature = (n_max, sigma))]
pub fn fubini_harmonic_spectrum(py: Python<'_>, n_max: u32, sigma: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = wave::fubini_harmonic_spectrum(n_max, sigma);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the shock formation distance for a finite-amplitude plane wave.
///
/// x_s = rho0 * c0^3 / (beta * omega * p0)
///
/// Args:
///     p0_pa: Source pressure amplitude [Pa].
///     f0_hz: Fundamental frequency [Hz].
///     c0: Small-signal sound speed [m/s].
///     rho0: Ambient density [kg/m³].
///     beta: Nonlinearity coefficient (1 + B/(2A)).
///
/// Returns:
///     Shock formation distance [m].
#[pyfunction]
#[pyo3(signature = (p0_pa, f0_hz, c0, rho0, beta))]
pub fn shock_formation_distance(
    p0_pa: f64,
    f0_hz: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
) -> PyResult<f64> {
    Ok(wave::shock_formation_distance(p0_pa, f0_hz, c0, rho0, beta))
}

/// Compute Westervelt harmonic pressure evolution along a propagation axis.
///
/// Returns a 2-D array of shape (len(z_arr), n_max) containing the pressure
/// amplitude [Pa] of each harmonic at each axial position.
///
/// Args:
///     z_arr: Axial positions [m].
///     p0: Source pressure amplitude [Pa].
///     f0: Fundamental frequency [Hz].
///     c0: Sound speed [m/s].
///     rho0: Density [kg/m³].
///     beta: Nonlinearity coefficient.
///     alpha_np_m: Absorption at f0 [Np/m].
///     n_max: Number of harmonics to compute.
///
/// Returns:
///     ndarray of shape (nz, n_max).
#[pyfunction]
#[pyo3(signature = (z_arr, p0, f0, c0, rho0, beta, alpha_np_m, n_max))]
pub fn westervelt_harmonic_evolution(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    p0: f64,
    f0: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
    alpha_np_m: f64,
    n_max: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let z_slice = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rows: Vec<Vec<f64>> =
        wave::westervelt_harmonic_evolution(z_slice, p0, f0, c0, rho0, beta, alpha_np_m, n_max);
    let nz = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let arr2d = Array2::from_shape_vec((nz, n_max), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}
