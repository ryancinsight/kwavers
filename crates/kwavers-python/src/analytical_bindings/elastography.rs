//! PyO3 bindings for `kwavers_physics::analytical::elastography`.

mod thermal_strain;

use kwavers_physics::analytical::elastography;
use ndarray::Array2;
use numpy::{ToPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub use thermal_strain::{
    thermal_strain_combined_coefficient, thermal_strain_reconstruct, thermal_strain_rf_fixture,
};

/// Compute the shear-wave speed from shear modulus and density.
///
/// c_s = sqrt(mu / rho)
///
/// Args:
///     mu_pa: Shear modulus [Pa].
///     rho: Density [kg/m³].
///
/// Returns:
///     Shear-wave speed [m/s].
#[pyfunction]
#[pyo3(signature = (mu_pa, rho))]
pub fn shear_wave_speed(mu_pa: f64, rho: f64) -> PyResult<f64> {
    Ok(elastography::shear_wave_speed(mu_pa, rho))
}

/// Ratio of compressional (P) to shear (S) wave speed from Poisson's ratio.
///
/// c_P/c_S = sqrt(2(1-nu)/(1-2nu)) (book §11.2); diverges as nu -> 1/2.
///
/// Args:
///     poisson_ratio: Poisson's ratio nu (dimensionless, < 0.5).
///
/// Returns:
///     c_P/c_S ratio (dimensionless; inf for nu >= 0.5).
#[pyfunction]
#[pyo3(signature = (poisson_ratio))]
pub fn pwave_to_swave_velocity_ratio(poisson_ratio: f64) -> PyResult<f64> {
    Ok(elastography::pwave_to_swave_velocity_ratio(poisson_ratio))
}

/// Compute the Voigt complex shear modulus G*(ω).
///
/// G*(ω) = mu + i*omega*eta
///
/// Args:
///     omega_arr: Angular frequency array [rad/s].
///     mu_pa: Elastic modulus [Pa].
///     eta_pa_s: Viscosity [Pa·s].
///
/// Returns:
///     (real_part, imag_part) tuple of arrays [Pa].
#[pyfunction]
#[pyo3(signature = (omega_arr, mu_pa, eta_pa_s))]
pub fn voigt_complex_modulus(
    py: Python<'_>,
    omega_arr: PyReadonlyArray1<f64>,
    mu_pa: f64,
    eta_pa_s: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let om_s = omega_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = elastography::voigt_complex_modulus(om_s, mu_pa, eta_pa_s);
    let real: Vec<f64> = result.iter().map(|c| c.re).collect();
    let imag: Vec<f64> = result.iter().map(|c| c.im).collect();
    Ok((
        real.to_pyarray(py).unbind(),
        imag.to_pyarray(py).unbind(),
    ))
}

/// Compute the springpot (fractional Kelvin) complex shear modulus.
///
/// G*(ω) = G0 * (i*ω)^alpha
///
/// Args:
///     omega_arr: Angular frequency array [rad/s].
///     g0: Quasi-static modulus scale [Pa·s^alpha].
///     alpha_exp: Fractional exponent in [0, 1].
///
/// Returns:
///     (real_part, imag_part) tuple of arrays [Pa].
#[pyfunction]
#[pyo3(signature = (omega_arr, g0, alpha_exp))]
pub fn springpot_complex_modulus(
    py: Python<'_>,
    omega_arr: PyReadonlyArray1<f64>,
    g0: f64,
    alpha_exp: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let om_s = omega_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = elastography::springpot_complex_modulus(om_s, g0, alpha_exp);
    let real: Vec<f64> = result.iter().map(|c| c.re).collect();
    let imag: Vec<f64> = result.iter().map(|c| c.im).collect();
    Ok((
        real.to_pyarray(py).unbind(),
        imag.to_pyarray(py).unbind(),
    ))
}

/// Compute the Voigt shear-wave phase velocity dispersion curve.
///
/// Args:
///     f_arr: Frequency array [Hz].
///     mu_pa: Shear modulus [Pa].
///     eta_pa_s: Viscosity [Pa·s].
///     rho: Density [kg/m³].
///
/// Returns:
///     Phase velocity array [m/s].
#[pyfunction]
#[pyo3(signature = (f_arr, mu_pa, eta_pa_s, rho))]
pub fn voigt_shear_wave_dispersion(
    py: Python<'_>,
    f_arr: PyReadonlyArray1<f64>,
    mu_pa: f64,
    eta_pa_s: f64,
    rho: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = elastography::voigt_shear_wave_dispersion(f_s, mu_pa, eta_pa_s, rho);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the 2-D MRE displacement field for a harmonic shear wave.
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     shear_speed: Shear-wave phase velocity [m/s].
///     freq_hz: Excitation frequency [Hz].
///     amplitude: Displacement amplitude [m].
///     penetration_depth_m: Exponential decay length [m].
///
/// Returns:
///     Displacement field ndarray of shape (len(x_arr), len(z_arr)) [m].
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, shear_speed, freq_hz, amplitude, penetration_depth_m))]
pub fn mre_displacement_field(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    shear_speed: f64,
    freq_hz: f64,
    amplitude: f64,
    penetration_depth_m: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let flat = elastography::mre_displacement_field(
        x_s,
        z_s,
        shear_speed,
        freq_hz,
        amplitude,
        penetration_depth_m,
    );
    let arr2d = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.to_pyarray(py).unbind())
}

/// Compute the positive exponential MRE displacement envelope.
#[pyfunction]
#[pyo3(signature = (z_arr, amplitude_m, penetration_depth_m))]
pub fn mre_displacement_envelope(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    amplitude_m: f64,
    penetration_depth_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let envelope = elastography::mre_displacement_envelope(z_s, amplitude_m, penetration_depth_m)
        .map_err(PyValueError::new_err)?;
    Ok(envelope.to_pyarray(py).unbind())
}

