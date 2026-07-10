//! PyO3 wrappers for 2-D transducer beam and focus helpers.

use kwavers_physics::analytical::transducer;
use numpy::ndarray::Array2;
use numpy::{ToPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute time-delay laws for 2-D geometric focusing.
///
/// Args:
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     x_f: Focal point x-coordinate [m].
///     z_f: Focal point z-coordinate [m].
///     c: Sound speed [m/s].
///
/// Returns:
///     Delay array [s], same length as *elem_x*.
#[pyfunction]
#[pyo3(signature = (elem_x, elem_z, x_f, z_f, c))]
pub fn delay_law_focus_2d(
    py: Python<'_>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    x_f: f64,
    z_f: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::delay_law_focus_2d(ex, ez, x_f, z_f, c);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the complex 2-D beam pattern for a phased array.
///
/// Returns a list [real_field, imag_field], each a 2-D ndarray of shape
/// (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///     weights: Apodization weights per element.
///     delays: Delay per element [s].
///
/// Returns:
///     [real_nx_nz, imag_nx_nz] — list of two 2-D arrays.
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, elem_x, elem_z, freq_hz, c, weights, delays))]
pub fn beam_pattern_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    freq_hz: f64,
    c: f64,
    weights: PyReadonlyArray1<f64>,
    delays: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let w_s = weights
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d_s = delays
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let (real_flat, imag_flat) =
        transducer::beam_pattern_2d(x_s, z_s, ex, ez, freq_hz, c, w_s, d_s);
    let real_arr = Array2::from_shape_vec((nx, nz), real_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let imag_arr = Array2::from_shape_vec((nx, nz), imag_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        real_arr.to_pyarray(py).unbind(),
        imag_arr.to_pyarray(py).unbind(),
    ))
}

/// Full far-field beam-pattern magnitude via the pattern-multiplication theorem.
///
/// |P(θ)| = |D(θ)| · |AF(θ)|, normalised to unit peak across the angle set.
///
/// Args:
///     theta_rad: Observation angles [rad].
///     k: Wave number [rad/m].
///     d_m: Element pitch [m].
///     n: Number of elements.
///     steer_rad: Steering angle [rad].
///     ka_elem: Element directivity parameter k·a_elem.
///
/// Returns:
///     Normalised pattern magnitude (linear), peak = 1.
#[pyfunction]
#[pyo3(signature = (theta_rad, k, d_m, n, steer_rad, ka_elem))]
pub fn beam_pattern_magnitude(
    py: Python<'_>,
    theta_rad: PyReadonlyArray1<f64>,
    k: f64,
    d_m: f64,
    n: usize,
    steer_rad: f64,
    ka_elem: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_slice = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::beam_pattern_magnitude(t_slice, k, d_m, n, steer_rad, ka_elem);
    Ok(result.to_pyarray(py).unbind())
}

/// Magnitude of the 2-D complex beam pattern, normalised to its peak.
///
/// Returns |p(x, z)| / max|p| as a 2-D ndarray of shape (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///     weights: Apodization weights per element.
///     delays: Delay per element [s].
///
/// Returns:
///     2-D magnitude field in [0, 1].
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, elem_x, elem_z, freq_hz, c, weights, delays))]
#[allow(clippy::too_many_arguments)]
pub fn beam_pattern_2d_magnitude(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    freq_hz: f64,
    c: f64,
    weights: PyReadonlyArray1<f64>,
    delays: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let w_s = weights
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d_s = delays
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let flat = transducer::beam_pattern_2d_magnitude(x_s, z_s, ex, ez, freq_hz, c, w_s, d_s);
    let arr = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.to_pyarray(py).unbind())
}

