//! Standing-wave suppression and modulation bindings.

use kwavers_physics::analytical::rtm as rtm_mod;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute temporal modulation frequencies for transcranial standing-wave suppression.
///
/// Args:
///     f0_hz: Carrier frequency `Hz`.
///     m_steps: Number of modulation steps.
///     c: Sound speed [m/s].
///     d_back_m: Back-wall distance `m`.
///
/// Returns:
///     Modulation frequency array `Hz`.
#[pyfunction]
#[pyo3(signature = (f0_hz, m_steps, c, d_back_m))]
pub fn temporal_modulation_frequencies(
    py: Python<'_>,
    f0_hz: f64,
    m_steps: usize,
    c: f64,
    d_back_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = rtm_mod::temporal_modulation_frequencies(f0_hz, m_steps, c, d_back_m);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the axial spatial frequency of the standing-wave pattern [cycles/m].
///
/// k_sw = 2·f / c   (= 1 / half-wavelength)
///
/// Used to locate the expected FFT peak in an axial spatial-frequency spectrum.
///
/// Args:
///     freq_hz: Frequency `Hz`.
///     c: Sound speed [m/s].
///
/// Returns:
///     Spatial frequency [cycles/m].
#[pyfunction]
#[pyo3(signature = (freq_hz, c))]
pub fn standing_wave_spatial_frequency_cycles_m(freq_hz: f64, c: f64) -> PyResult<f64> {
    Ok(rtm_mod::standing_wave_spatial_frequency_cycles_m(
        freq_hz, c,
    ))
}

/// Compute the standing-wave suppression gain factor.
///
/// G = (1 + R)² / (1 + R²)
///
/// Args:
///     r_back: Back-wall pressure reflection coefficient magnitude.
///
/// Returns:
///     Suppression gain factor (≥ 1).
#[pyfunction]
#[pyo3(signature = (r_back,))]
pub fn standing_wave_suppression_gain(r_back: f64) -> PyResult<f64> {
    Ok(rtm_mod::standing_wave_suppression_gain(r_back))
}

/// Compute the period of one full standing-wave modulation cycle `Hz`.
///
/// ΔF_period = c / (2 · d_back_m)
///
/// Args:
///     c: Sound speed [m/s].
///     d_back_m: Distance from field point to back-reflecting wall `m`.
///
/// Returns:
///     Modulation period in frequency units `Hz`.
#[pyfunction]
#[pyo3(signature = (c, d_back_m))]
pub fn standing_wave_modulation_period_hz(c: f64, d_back_m: f64) -> PyResult<f64> {
    Ok(rtm_mod::standing_wave_modulation_period_hz(c, d_back_m))
}

/// Compute the 1-D standing-wave intensity pattern: |1 + R·exp(2ik·x)|².
///
/// Exact result for a plane wave in a lossless 1-D medium:
///   SW²(x) = 1 + R² + 2R·cos(2kx),   k = 2π·f/c
///
/// Args:
///     x_arr: Distances from the back reflector `m`.
///     freq_hz: Frequency `Hz`.
///     c: Sound speed [m/s].
///     r_back: Pressure reflection coefficient.
///
/// Returns:
///     SW² array of same length as x_arr.
#[pyfunction]
#[pyo3(signature = (x_arr, freq_hz, c, r_back))]
pub fn standing_wave_field_1d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    freq_hz: f64,
    c: f64,
    r_back: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = rtm_mod::standing_wave_field_1d(x_s, freq_hz, c, r_back);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute exact statistical moments of the standing-wave intensity pattern.
///
/// Returns (sw2_mean, sw2_peak, sw2_trough):
///   sw2_mean   = 1 + R²          (spatial average / RTM-modulated mean)
///   sw2_peak   = (1 + R)²        (antinodal maximum)
///   sw2_trough = (1 − R)²        (nodal minimum)
///
/// Args:
///     r_back: Pressure reflection coefficient.
///
/// Returns:
///     (sw2_mean, sw2_peak, sw2_trough) tuple of floats.
#[pyfunction]
#[pyo3(signature = (r_back,))]
pub fn standing_wave_intensity_statistics(r_back: f64) -> PyResult<(f64, f64, f64)> {
    Ok(rtm_mod::standing_wave_intensity_statistics(r_back))
}
