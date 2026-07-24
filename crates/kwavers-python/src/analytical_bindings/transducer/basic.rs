//! PyO3 wrappers for basic transducer directivity and apodization helpers.

use kwavers_physics::analytical::transducer;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Compute the circular-piston far-field directivity pattern.
///
/// D(theta) = 2 * J1(ka*sin(theta)) / (ka*sin(theta))
///
/// Args:
///     theta_rad: Observation angles `rad`.
///     ka: Wave-number × radius product.
///
/// Returns:
///     Directivity array (normalised to unity on-axis).
#[pyfunction]
#[pyo3(signature = (theta_rad, ka))]
pub fn circular_piston_directivity(
    py: Python<'_>,
    theta_rad: PyReadonlyArray1<f64>,
    ka: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_slice = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::circular_piston_directivity(t_slice, ka);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the linear-array factor as a function of angle.
///
/// Args:
///     theta_rad: Observation angles `rad`.
///     k: Wave number [rad/m].
///     d_m: Element pitch `m`.
///     n: Number of elements.
///     steer_rad: Electronic steering angle `rad`.
///
/// Returns:
///     Normalised array factor.
#[pyfunction]
#[pyo3(signature = (theta_rad, k, d_m, n, steer_rad))]
pub fn linear_array_factor(
    py: Python<'_>,
    theta_rad: PyReadonlyArray1<f64>,
    k: f64,
    d_m: f64,
    n: usize,
    steer_rad: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_slice = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::linear_array_factor(t_slice, k, d_m, n, steer_rad);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute grating-lobe angles for a uniform linear array.
///
/// Args:
///     k: Wave number [rad/m].
///     d_m: Element pitch `m`.
///     steer_rad: Steering angle `rad`.
///
/// Returns:
///     Array of grating-lobe angles `rad` (may be empty if none exist).
#[pyfunction]
#[pyo3(signature = (k, d_m, steer_rad))]
pub fn grating_lobe_angles(
    py: Python<'_>,
    k: f64,
    d_m: f64,
    steer_rad: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = transducer::grating_lobe_angles(k, d_m, steer_rad);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute element apodization weights for a given window type.
///
/// Supported window types: "uniform", "hann", "hamming", "blackman",
/// "nuttall", "tukey".
///
/// Args:
///     n: Number of elements.
///     window_type: Name of the window function.
///
/// Returns:
///     Weight array of length *n*.
#[pyfunction]
#[pyo3(signature = (n, window_type))]
pub fn apodization_weights(
    py: Python<'_>,
    n: usize,
    window_type: String,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = transducer::apodization_weights(n, &window_type);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute apodization weights and normalized FFT-shifted response.
///
/// Args:
///     n_elements: Number of array elements.
///     window_type: Window name accepted by `apodization_weights`.
///     nfft: Zero-padded FFT length.
///
/// Returns:
///     `(weights, cycles_per_aperture, response_db)`.
#[pyfunction]
#[pyo3(signature = (n_elements, window_type, nfft))]
pub fn apodization_window_response(
    py: Python<'_>,
    n_elements: usize,
    window_type: String,
    nfft: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let response = transducer::apodization_window_response(n_elements, &window_type, nfft)
        .map_err(PyValueError::new_err)?;
    Ok((
        response.weights.to_pyarray(py).unbind(),
        response.cycles_per_aperture.to_pyarray(py).unbind(),
        response.response_db.to_pyarray(py).unbind(),
    ))
}

/// Compute the on-axis pressure of a circular piston transducer.
///
/// Args:
///     z_arr: Axial positions `m`.
///     radius_m: Piston radius `m`.
///     freq_hz: Frequency `Hz`.
///     p0_pa: Surface pressure amplitude `Pa`.
///     c: Sound speed [m/s].
///
/// Returns:
///     On-axis pressure magnitude `Pa`.
#[pyfunction]
#[pyo3(signature = (z_arr, radius_m, freq_hz, p0_pa, c))]
pub fn circular_piston_onaxis(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    radius_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::circular_piston_onaxis(z_s, radius_m, freq_hz, p0_pa, c);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the on-axis pressure of a focused-bowl (spherically focused) transducer.
///
/// Args:
///     z_arr: Axial positions `m`.
///     bowl_radius_m: Bowl aperture radius `m`.
///     focal_length_m: Geometric focal length `m`.
///     freq_hz: Frequency `Hz`.
///     p0_pa: Source pressure `Pa`.
///     c: Sound speed [m/s].
///
/// Returns:
///     On-axis pressure magnitude `Pa`.
#[pyfunction]
#[pyo3(signature = (z_arr, bowl_radius_m, focal_length_m, freq_hz, p0_pa, c))]
pub fn focused_bowl_onaxis(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    bowl_radius_m: f64,
    focal_length_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        transducer::focused_bowl_onaxis(z_s, bowl_radius_m, focal_length_m, freq_hz, p0_pa, c);
    Ok(result.to_pyarray(py).unbind())
}
