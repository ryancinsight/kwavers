//! PyO3 bindings for `kwavers::physics::analytical::rtm`.

use kwavers::physics::analytical::rtm as rtm_mod;
use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute a focused Gaussian beam field in 2-D including skull transmission.
///
/// Returns (real_field, imag_field) each of shape (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     x_f: Focus x-coordinate [m].
///     z_f: Focus z-coordinate [m].
///     freq_hz: Frequency [Hz].
///     c_brain: Brain sound speed [m/s].
///     w0_m: Beam waist [m].
///     skull_transmission_real: Real part of skull transmission coefficient.
///     skull_transmission_imag: Imaginary part.
///     r_back: Back-wall reflection coefficient.
///     z_back: Back-wall axial position [m].
///
/// Returns:
///     (real_nx_nz, imag_nx_nz) tuple of 2-D arrays.
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, x_f, z_f, freq_hz, c_brain, w0_m, skull_transmission_real, skull_transmission_imag, r_back, z_back))]
pub fn focused_gaussian_beam_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    x_f: f64,
    z_f: f64,
    freq_hz: f64,
    c_brain: f64,
    w0_m: f64,
    skull_transmission_real: f64,
    skull_transmission_imag: f64,
    r_back: f64,
    z_back: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let skull_transmission = Complex64::new(skull_transmission_real, skull_transmission_imag);
    let (real_flat, imag_flat) = rtm_mod::focused_gaussian_beam_2d(
        x_s,
        z_s,
        x_f,
        z_f,
        freq_hz,
        c_brain,
        w0_m,
        skull_transmission,
        r_back,
        z_back,
    );
    let real_arr = Array2::from_shape_vec((nx, nz), real_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let imag_arr = Array2::from_shape_vec((nx, nz), imag_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        real_arr.into_pyarray(py).unbind(),
        imag_arr.into_pyarray(py).unbind(),
    ))
}

/// Compute the 2-D back-propagation Green's function.
///
/// Returns (real_field, imag_field) each of shape (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     x_f: Source x-coordinate [m].
///     z_f: Source z-coordinate [m].
///     k_br: Wave number in brain [rad/m].
///
/// Returns:
///     (real_nx_nz, imag_nx_nz) tuple.
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, x_f, z_f, k_br))]
pub fn backprop_green_function_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    x_f: f64,
    z_f: f64,
    k_br: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let (real_flat, imag_flat) = rtm_mod::backprop_green_function_2d(x_s, z_s, x_f, z_f, k_br);
    let real_arr = Array2::from_shape_vec((nx, nz), real_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let imag_arr = Array2::from_shape_vec((nx, nz), imag_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        real_arr.into_pyarray(py).unbind(),
        imag_arr.into_pyarray(py).unbind(),
    ))
}

/// Apply the zero-lag cross-correlation imaging condition for RTM.
///
/// image[i,j] = sum_t p_fwd[i,j,t] * p_bwd[i,j,t]
///
/// This version operates on single-frequency snapshots (no time axis):
/// image[i,j] = Re(p_fwd[i,j] * conj(p_bwd[i,j]))
///
/// Args:
///     p_fwd_real: Real part of forward-propagated field (nx × nz).
///     p_fwd_imag: Imaginary part.
///     p_bwd_real: Real part of back-propagated field (nx × nz).
///     p_bwd_imag: Imaginary part.
///     nx: Number of lateral grid points.
///     nz: Number of axial grid points.
///
/// Returns:
///     Image ndarray of shape (nx, nz).
#[pyfunction]
#[pyo3(signature = (p_fwd_real, p_fwd_imag, p_bwd_real, p_bwd_imag, nx, nz))]
pub fn rtm_imaging_condition(
    py: Python<'_>,
    p_fwd_real: PyReadonlyArray2<f64>,
    p_fwd_imag: PyReadonlyArray2<f64>,
    p_bwd_real: PyReadonlyArray2<f64>,
    p_bwd_imag: PyReadonlyArray2<f64>,
    nx: usize,
    nz: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let fr = p_fwd_real.as_array();
    let fi = p_fwd_imag.as_array();
    let br = p_bwd_real.as_array();
    let bi = p_bwd_imag.as_array();
    let fr_flat: Vec<f64> = (0..nx)
        .flat_map(|i| (0..nz).map(move |j| fr[[i, j]]))
        .collect();
    let fi_flat: Vec<f64> = (0..nx)
        .flat_map(|i| (0..nz).map(move |j| fi[[i, j]]))
        .collect();
    let br_flat: Vec<f64> = (0..nx)
        .flat_map(|i| (0..nz).map(move |j| br[[i, j]]))
        .collect();
    let bi_flat: Vec<f64> = (0..nx)
        .flat_map(|i| (0..nz).map(move |j| bi[[i, j]]))
        .collect();
    let flat = rtm_mod::rtm_imaging_condition(&fr_flat, &fi_flat, &br_flat, &bi_flat, nx, nz);
    let arr2d = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

/// Fuse multiple single-frequency RTM images by coherent averaging.
///
/// Args:
///     images: List of PyReadonlyArray2 images, each of shape (nx, nz).
///
/// Returns:
///     Fused image of shape (nx, nz).
#[pyfunction]
#[pyo3(signature = (images,))]
pub fn rtm_multi_frequency_fusion(
    py: Python<'_>,
    images: Vec<PyReadonlyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    if images.is_empty() {
        return Err(PyRuntimeError::new_err("images list must not be empty"));
    }
    let first = images[0].as_array();
    let (nx, nz) = first.dim();
    let vecs: Vec<Vec<f64>> = images
        .iter()
        .map(|img| {
            let arr = img.as_array();
            (0..nx)
                .flat_map(|i| (0..nz).map(move |j| arr[[i, j]]))
                .collect()
        })
        .collect();
    let flat = rtm_mod::rtm_multi_frequency_fusion(&vecs);
    let arr2d = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

/// Compute temporal modulation frequencies for transcranial standing-wave suppression.
///
/// Args:
///     f0_hz: Carrier frequency [Hz].
///     m_steps: Number of modulation steps.
///     c: Sound speed [m/s].
///     d_back_m: Back-wall distance [m].
///
/// Returns:
///     Modulation frequency array [Hz].
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
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the standing-wave suppression gain factor.
///
/// G = 1 - |r_back|²
///
/// Args:
///     r_back: Back-wall reflection coefficient magnitude.
///
/// Returns:
///     Suppression gain factor (0–1).
#[pyfunction]
#[pyo3(signature = (r_back,))]
pub fn standing_wave_suppression_gain(r_back: f64) -> PyResult<f64> {
    Ok(rtm_mod::standing_wave_suppression_gain(r_back))
}
