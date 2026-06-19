//! PyO3 bindings for `kwavers_physics::analytical::elastography`.

use kwavers_physics::acoustics::imaging::modalities::elastography::thermal_strain::TrackingParams;
use kwavers_physics::acoustics::imaging::modalities::elastography::{
    ThermalStrainConfig, ThermalStrainImager,
};
use kwavers_physics::analytical::elastography;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
        real.into_pyarray(py).unbind(),
        imag.into_pyarray(py).unbind(),
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
        real.into_pyarray(py).unbind(),
        imag.into_pyarray(py).unbind(),
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
    Ok(result.into_pyarray(py).unbind())
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
    Ok(arr2d.into_pyarray(py).unbind())
}

/// Combined thermoacoustic strain coefficient `k_T = α_th − (1/c₀)·(dc/dT)` [1/°C]
/// — the apparent thermal strain per unit temperature change (book §11.12).
///
/// Delegates to `ThermalStrainConfig::combined_coefficient` (Rust SSOT).
///
/// Args:
///     sound_speed: Reference sound speed c₀ [m/s].
///     dc_dt: Temperature coefficient of sound speed dc/dT [m/s per °C].
///     thermal_expansion: Linear thermal-expansion coefficient α_th [1/°C].
///
/// Returns:
///     k_T [1/°C] (negative for water-based tissue, positive for lipid).
#[pyfunction]
#[pyo3(signature = (sound_speed, dc_dt, thermal_expansion))]
pub fn thermal_strain_combined_coefficient(
    sound_speed: f64,
    dc_dt: f64,
    thermal_expansion: f64,
) -> f64 {
    ThermalStrainConfig {
        sound_speed,
        dc_dt,
        thermal_expansion,
        strain_window: 11,
    }
    .combined_coefficient()
}

/// Reconstruct (displacement, thermal strain, temperature change) from a pre- and
/// post-heating RF volume via the `ThermalStrainImager` pipeline (book §11.12):
/// NCC axial tracking → least-squares strain → `ΔT = ε_T / k_T`.
///
/// Both volumes are `[nx, ny, nz]` with the axial (fast-time) direction along the
/// last axis. Physics lives entirely in the Rust core; the caller supplies only
/// the synthetic/measured RF and the acquisition parameters.
///
/// Args:
///     reference: Pre-heating RF volume `[nx, ny, nz]`.
///     tracked: Post-heating RF volume `[nx, ny, nz]`.
///     sound_speed: Reference sound speed c₀ [m/s].
///     dc_dt: dc/dT [m/s per °C].
///     thermal_expansion: α_th [1/°C].
///     strain_window: Odd least-squares strain window length (≥ 3).
///     sampling_rate: RF sampling rate f_s [Hz] (Δz = c₀/(2 f_s)).
///     window_half: NCC correlation kernel half-length [samples].
///     max_lag: NCC maximum search lag [samples].
///
/// Returns:
///     `(displacement_m, strain, temperature_change_c)`, each `[nx, ny, nz]`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    reference, tracked, sound_speed, dc_dt, thermal_expansion,
    strain_window, sampling_rate, window_half, max_lag
))]
pub fn thermal_strain_reconstruct(
    py: Python<'_>,
    reference: PyReadonlyArray3<f64>,
    tracked: PyReadonlyArray3<f64>,
    sound_speed: f64,
    dc_dt: f64,
    thermal_expansion: f64,
    strain_window: usize,
    sampling_rate: f64,
    window_half: usize,
    max_lag: usize,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray3<f64>>, Py<PyArray3<f64>>)> {
    let config = ThermalStrainConfig {
        sound_speed,
        dc_dt,
        thermal_expansion,
        strain_window,
    };
    let tracking = TrackingParams {
        window_half,
        max_lag,
    };
    let imager = ThermalStrainImager::new(config, tracking, sampling_rate)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let reference = reference.as_array().to_owned();
    let tracked = tracked.as_array().to_owned();
    let result = imager
        .reconstruct_temperature(&reference, &tracked)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        result.displacement.into_pyarray(py).unbind(),
        result.strain.into_pyarray(py).unbind(),
        result.temperature_change.into_pyarray(py).unbind(),
    ))
}
