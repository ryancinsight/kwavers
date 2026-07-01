//! RTM field synthesis and back-propagation bindings.

use super::arrays::complex_field_arrays;
use kwavers_physics::analytical::rtm as rtm_mod;
use num_complex::Complex64;
use numpy::{PyArray2, PyReadonlyArray1};
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
    complex_field_arrays(py, x_s.len(), z_s.len(), real_flat, imag_flat)
}

/// Compute the 2-D back-propagation Green's function.
///
/// Exact solution of the 2-D Helmholtz equation for a point source at
/// (x_f, z_f) in a homogeneous medium:
///   P_bwd(x,z) = exp(−ik·r_f) / √r_f,  k = 2π·f/c
///
/// Returns (real_field, imag_field) each of shape (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     x_f: Source x-coordinate [m].
///     z_f: Source z-coordinate [m].
///     freq_hz: Frequency [Hz].
///     c: Sound speed in the coupling medium [m/s].
///
/// Returns:
///     (real_nx_nz, imag_nx_nz) tuple.
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, x_f, z_f, freq_hz, c))]
pub fn backprop_green_function_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    x_f: f64,
    z_f: f64,
    freq_hz: f64,
    c: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (real_flat, imag_flat) =
        rtm_mod::backprop_green_function_2d(x_s, z_s, x_f, z_f, freq_hz, c);
    complex_field_arrays(py, x_s.len(), z_s.len(), real_flat, imag_flat)
}
