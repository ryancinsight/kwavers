//! Point-spread and resolution imaging bindings.

use kwavers_physics::analytical::imaging;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the lateral point-spread function using a sinc² model.
///
/// Args:
///     x_arr: Lateral positions [m].
///     f_number: F-number (focal length / aperture).
///     wavelength_m: Acoustic wavelength [m].
///
/// Returns:
///     Normalised lateral PSF array.
#[pyfunction]
#[pyo3(signature = (x_arr, f_number, wavelength_m))]
pub fn lateral_psf_sinc2(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    f_number: f64,
    wavelength_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = imaging::lateral_psf_sinc2(x_s, f_number, wavelength_m);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the axial point-spread function using a rectangular-spectrum model.
///
/// Args:
///     z_arr: Axial positions [m].
///     c: Sound speed [m/s].
///     bandwidth_hz: Transducer −6 dB bandwidth [Hz].
///
/// Returns:
///     Normalised axial PSF array.
#[pyfunction]
#[pyo3(signature = (z_arr, c, bandwidth_hz))]
pub fn axial_psf_rect(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    c: f64,
    bandwidth_hz: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = imaging::axial_psf_rect(z_s, c, bandwidth_hz);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the plane-wave compounding lateral PSF.
///
/// Args:
///     x_arr: Lateral positions [m].
///     n_angles: Number of compounding angles.
///     f_number: F-number.
///     wavelength_m: Wavelength [m].
///
/// Returns:
///     Normalised compounded lateral PSF array.
#[pyfunction]
#[pyo3(signature = (x_arr, n_angles, f_number, wavelength_m))]
pub fn pw_compounding_lateral_psf(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    n_angles: usize,
    f_number: f64,
    wavelength_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = imaging::pw_compounding_lateral_psf(x_s, n_angles, f_number, wavelength_m);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the −6 dB lateral resolution.
///
/// δx ≈ f_number * wavelength
///
/// Args:
///     f_number: F-number.
///     wavelength_m: Wavelength [m].
///
/// Returns:
///     Lateral resolution [m].
#[pyfunction]
#[pyo3(signature = (f_number, wavelength_m))]
pub fn lateral_resolution_m(f_number: f64, wavelength_m: f64) -> PyResult<f64> {
    Ok(imaging::lateral_resolution_m(f_number, wavelength_m))
}

