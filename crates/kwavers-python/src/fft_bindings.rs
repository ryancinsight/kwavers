//! Python bindings for kwavers FFT utilities.
//!
//! Wraps the apollo-backed 1-D and 3-D real↔complex DFT functions exported
//! under `kwavers::math::fft`.  The forward transforms accept f64 real arrays
//! and return complex128 arrays; the inverse transforms accept complex128 and
//! return f64.

use kwavers::math::fft::{fft_1d_array, fft_3d_array, ifft_1d_array, ifft_3d_array};
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Forward 1-D DFT of a real-valued signal.
///
/// # Parameters
/// - `signal`: 1-D array of `f64` samples.
///
/// # Returns
/// 1-D array of `complex128` with the same length as `signal`, containing
/// the full (two-sided) discrete Fourier spectrum normalised by 1/N.
#[pyfunction]
pub fn fft1<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    let arr: Array1<f64> = signal.as_array().to_owned();
    if arr.is_empty() {
        return Err(PyValueError::new_err(
            "fft1: input signal must be non-empty",
        ));
    }
    let spectrum = py.detach(|| fft_1d_array(&arr));
    Ok(spectrum.into_pyarray(py).into())
}

/// Inverse 1-D DFT of a complex spectrum.
///
/// # Parameters
/// - `spectrum`: 1-D array of `complex128` (full two-sided spectrum).
///
/// # Returns
/// 1-D array of `f64` (real part of the IDFT output), length equal to the
/// length of `spectrum`.
#[pyfunction]
pub fn ifft1<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr: Array1<Complex64> = spectrum.as_array().to_owned();
    if arr.is_empty() {
        return Err(PyValueError::new_err(
            "ifft1: input spectrum must be non-empty",
        ));
    }
    let signal = py.detach(|| ifft_1d_array(&arr));
    Ok(signal.into_pyarray(py).into())
}

/// Forward 3-D DFT of a real-valued field.
///
/// # Parameters
/// - `field`: 3-D array of `f64` with shape `(nx, ny, nz)`.
///
/// # Returns
/// 3-D array of `complex128` with shape `(nx, ny, nz)` containing the full
/// (two-sided) discrete Fourier spectrum.
#[pyfunction]
pub fn fft3<'py>(
    py: Python<'py>,
    field: PyReadonlyArray3<'py, f64>,
) -> PyResult<Py<PyArray3<Complex64>>> {
    let arr: Array3<f64> = field.as_array().to_owned();
    let (nx, ny, nz) = arr.dim();
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(PyValueError::new_err(
            "fft3: all dimensions must be non-zero",
        ));
    }
    let spectrum = py.detach(|| fft_3d_array(&arr));
    Ok(spectrum.into_pyarray(py).into())
}

/// Inverse 3-D DFT of a complex spectrum.
///
/// # Parameters
/// - `spectrum`: 3-D array of `complex128` with shape `(nx, ny, nz)`.
///
/// # Returns
/// 3-D array of `f64` (real part of the IDFT output) with shape
/// `(nx, ny, nz)`.
#[pyfunction]
pub fn ifft3<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray3<'py, Complex64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let arr: Array3<Complex64> = spectrum.as_array().to_owned();
    let (nx, ny, nz) = arr.dim();
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(PyValueError::new_err(
            "ifft3: all dimensions must be non-zero",
        ));
    }
    let field = py.detach(|| ifft_3d_array(&arr));
    Ok(field.into_pyarray(py).into())
}

/// Register FFT binding functions into the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft1, m)?)?;
    m.add_function(wrap_pyfunction!(ifft1, m)?)?;
    m.add_function(wrap_pyfunction!(fft3, m)?)?;
    m.add_function(wrap_pyfunction!(ifft3, m)?)?;
    Ok(())
}
