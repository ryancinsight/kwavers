//! Skull attenuation and aberration bindings.

use kwavers_physics::analytical::skull as skull_mod;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute two-way skull insertion loss using a power-law attenuation model.
///
/// Args:
///     f_mhz: Frequency array [MHz].
///     thickness_cm: Skull thickness [cm].
///     alpha0: Attenuation coefficient [dB/(cm·MHz)].
///
/// Returns:
///     Two-way insertion loss array [dB].
#[pyfunction]
#[pyo3(signature = (f_mhz, thickness_cm, alpha0))]
pub fn skull_insertion_loss_two_way_db(
    py: Python<'_>,
    f_mhz: PyReadonlyArray1<f64>,
    thickness_cm: f64,
    alpha0: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::skull_insertion_loss_two_way_db(f_s, thickness_cm, alpha0);
    Ok(result.to_pyarray(py).unbind())
}

/// Generate a random phase screen modelling skull aberration.
///
/// Args:
///     n: Number of phase-screen points.
///     sigma_phi_rad: Phase standard deviation [rad].
///     seed: RNG seed for reproducibility.
///
/// Returns:
///     Phase array [rad] of length *n*.
#[pyfunction]
#[pyo3(signature = (n, sigma_phi_rad, seed))]
pub fn skull_phase_screen(
    py: Python<'_>,
    n: usize,
    sigma_phi_rad: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = skull_mod::skull_phase_screen(n, sigma_phi_rad, seed);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the Strehl ratio for a given wavefront-error standard deviation.
///
/// S ≈ exp(-sigma_phi²)  (Maréchal approximation)
///
/// Args:
///     sigma_phi_rad: RMS wavefront error [rad].
///
/// Returns:
///     Strehl ratio (0–1).
#[pyfunction]
#[pyo3(signature = (sigma_phi_rad,))]
pub fn strehl_ratio(sigma_phi_rad: f64) -> PyResult<f64> {
    Ok(skull_mod::strehl_ratio(sigma_phi_rad))
}
