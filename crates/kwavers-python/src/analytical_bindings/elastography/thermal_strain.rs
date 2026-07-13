//! Thermal-strain elastography bindings.

use crate::breast_fwi_bindings::complex_compat::{leto3_to_nd3, nd_to_leto3};
use kwavers_physics::acoustics::imaging::modalities::elastography::thermal_strain::TrackingParams;
use kwavers_physics::acoustics::imaging::modalities::elastography::{
    ThermalStrainConfig, ThermalStrainImager,
};
use kwavers_physics::analytical::elastography;
use numpy::ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Generate deterministic pre/post-heating RF volumes for thermal-strain validation.
///
/// Args:
///     n_lines: Number of independent lateral RF lines.
///     nz: Axial samples per line.
///     k_t: Combined thermal-strain coefficient [1/°C].
///     delta_t_c: Uniform temperature change [°C].
///     samples_per_carrier: RF samples per carrier cycle.
///     seed: Deterministic random seed.
///
/// Returns:
///     `(reference, tracked)` RF volumes, each shaped `(n_lines, 1, nz)`.
#[pyfunction]
#[pyo3(signature = (n_lines, nz, k_t, delta_t_c, samples_per_carrier, seed))]
pub fn thermal_strain_rf_fixture(
    py: Python<'_>,
    n_lines: usize,
    nz: usize,
    k_t: f64,
    delta_t_c: f64,
    samples_per_carrier: f64,
    seed: u64,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray3<f64>>)> {
    let (reference, tracked) = elastography::thermal_strain_rf_fixture(
        n_lines,
        nz,
        k_t,
        delta_t_c,
        samples_per_carrier,
        seed,
    )
    .map_err(PyValueError::new_err)?;
    let reference = Array3::from_shape_vec((n_lines, 1, nz), reference)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let tracked = Array3::from_shape_vec((n_lines, 1, nz), tracked)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        reference.to_pyarray(py).unbind(),
        tracked.to_pyarray(py).unbind(),
    ))
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
    let reference = nd_to_leto3(reference.as_array().to_owned());
    let tracked = nd_to_leto3(tracked.as_array().to_owned());
    let result = imager
        .reconstruct_temperature(&reference, &tracked)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        leto3_to_nd3(result.displacement).to_pyarray(py).unbind(),
        leto3_to_nd3(result.strain).to_pyarray(py).unbind(),
        leto3_to_nd3(result.temperature_change)
            .to_pyarray(py)
            .unbind(),
    ))
}
