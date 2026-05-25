//! PyO3 bindings for `kwavers::physics::analytical::transducer`.

use kwavers::physics::analytical::transducer;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the circular-piston far-field directivity pattern.
///
/// D(theta) = 2 * J1(ka*sin(theta)) / (ka*sin(theta))
///
/// Args:
///     theta_rad: Observation angles [rad].
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
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the linear-array factor as a function of angle.
///
/// Args:
///     theta_rad: Observation angles [rad].
///     k: Wave number [rad/m].
///     d_m: Element pitch [m].
///     n: Number of elements.
///     steer_rad: Electronic steering angle [rad].
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
    Ok(result.into_pyarray(py).unbind())
}

/// Compute grating-lobe angles for a uniform linear array.
///
/// Args:
///     k: Wave number [rad/m].
///     d_m: Element pitch [m].
///     steer_rad: Steering angle [rad].
///
/// Returns:
///     Array of grating-lobe angles [rad] (may be empty if none exist).
#[pyfunction]
#[pyo3(signature = (k, d_m, steer_rad))]
pub fn grating_lobe_angles(
    py: Python<'_>,
    k: f64,
    d_m: f64,
    steer_rad: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = transducer::grating_lobe_angles(k, d_m, steer_rad);
    Ok(result.into_pyarray(py).unbind())
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
    Ok(result.into_pyarray(py).unbind())
}

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
    Ok(result.into_pyarray(py).unbind())
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
        real_arr.into_pyarray(py).unbind(),
        imag_arr.into_pyarray(py).unbind(),
    ))
}

/// Compute the on-axis pressure of a circular piston transducer.
///
/// Args:
///     z_arr: Axial positions [m].
///     radius_m: Piston radius [m].
///     freq_hz: Frequency [Hz].
///     p0_pa: Surface pressure amplitude [Pa].
///     c: Sound speed [m/s].
///
/// Returns:
///     On-axis pressure magnitude [Pa].
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
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the on-axis pressure of a focused-bowl (spherically focused) transducer.
///
/// Args:
///     z_arr: Axial positions [m].
///     bowl_radius_m: Bowl aperture radius [m].
///     focal_length_m: Geometric focal length [m].
///     freq_hz: Frequency [Hz].
///     p0_pa: Source pressure [Pa].
///     c: Sound speed [m/s].
///
/// Returns:
///     On-axis pressure magnitude [Pa].
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
    Ok(result.into_pyarray(py).unbind())
}

/// Compute band-limited interpolation (BLI) stencil weights.
///
/// Args:
///     delta: Sub-sample offsets (1-D array).
///     n_stencil: Number of stencil points.
///
/// Returns:
///     2-D array of shape (len(delta), n_stencil).
#[pyfunction]
#[pyo3(signature = (delta, n_stencil))]
pub fn bli_stencil_weights(
    py: Python<'_>,
    delta: PyReadonlyArray1<f64>,
    n_stencil: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let d_s = delta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rows: Vec<Vec<f64>> = transducer::bli_stencil_weights(d_s, n_stencil);
    let n_delta = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let arr2d = Array2::from_shape_vec((n_delta, n_stencil), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}
