//! Python bindings for kwavers FFT utilities.
//!
//! Wraps the apollo-backed 1-D and 3-D real↔complex DFT functions exported
//! under `kwavers_math::fft`.  The forward transforms accept f64 real arrays
//! and return complex128 arrays; the inverse transforms accept complex128 and
//! return f64.

use crate::breast_fwi_bindings::complex_compat::{
    ec_to_nc1, ec_to_nc3, leto1_to_nd1, leto3_to_nd3, nc_to_ec1, nc_to_ec3, nd_to_leto1,
    nd_to_leto3,
};
use kwavers_math::{
    fft::{fft_1d_array, fft_3d_array, ifft_1d_array, ifft_3d_array},
    signal::window::hann,
};
use leto::{
    Array1,
    Array3,
};
use eunomia::Complex64;
use numpy::{ToPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
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
    let spectrum = py.detach(|| fft_1d_array(&nd_to_leto1(arr)));
    Ok(ec_to_nc1(leto1_to_nd1(spectrum)).to_pyarray(py).into())
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
    let signal = py.detach(|| ifft_1d_array(&nd_to_leto1(nc_to_ec1(arr))));
    Ok(leto1_to_nd1(signal).to_pyarray(py).into())
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
    let spectrum = py.detach(|| fft_3d_array(&nd_to_leto3(arr)));
    Ok(ec_to_nc3(leto3_to_nd3(spectrum)).to_pyarray(py).into())
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
    let field = py.detach(|| ifft_3d_array(&nd_to_leto3(nc_to_ec3(arr))));
    Ok(leto3_to_nd3(field).to_pyarray(py).into())
}

/// Demeaned Hann-windowed one-sided power spectrum of a 1-D real profile.
///
/// The returned frequency axis matches NumPy's `rfftfreq(n, d=sample_spacing)`.
/// Power is `|FFT((x - mean(x)) * hann)|^2` using the same unnormalised forward
/// DFT as [`fft1`], so it is numerically equivalent to `np.abs(np.fft.rfft(...))**2`
/// for the one-sided bins.
#[pyfunction]
pub fn demeaned_hann_power_spectrum_1d<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    sample_spacing: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let input = signal.as_array();
    let n = input.len();
    if n < 2 {
        return Err(PyValueError::new_err(
            "demeaned_hann_power_spectrum_1d: signal length must be at least two",
        ));
    }
    if !(sample_spacing.is_finite() && sample_spacing > 0.0) {
        return Err(PyValueError::new_err(
            "demeaned_hann_power_spectrum_1d: sample_spacing must be positive and finite",
        ));
    }
    if !input.iter().all(|sample| sample.is_finite()) {
        return Err(PyValueError::new_err(
            "demeaned_hann_power_spectrum_1d: signal samples must be finite",
        ));
    }

    let mean = input.iter().sum::<f64>() / n as f64;
    let denominator = n as f64 - 1.0;
    let mut windowed = Vec::with_capacity(n);
    for (idx, &sample) in input.iter().enumerate() {
        windowed.push((sample - mean) * hann(idx as f64 / denominator));
    }

    let spectrum = py.detach(|| fft_1d_array(&nd_to_leto1(Array1::from_vec(windowed))));
    let one_sided = n / 2 + 1;
    let scale = 1.0 / (n as f64 * sample_spacing);
    let mut frequency = Vec::with_capacity(one_sided);
    let mut power = Vec::with_capacity(one_sided);
    for idx in 0..one_sided {
        frequency.push(idx as f64 * scale);
        power.push(spectrum[idx].norm_sqr());
    }

    Ok((
        Array1::from_vec(frequency).to_pyarray(py).into(),
        Array1::from_vec(power).to_pyarray(py).into(),
    ))
}

/// Register FFT binding functions into the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft1, m)?)?;
    m.add_function(wrap_pyfunction!(ifft1, m)?)?;
    m.add_function(wrap_pyfunction!(fft3, m)?)?;
    m.add_function(wrap_pyfunction!(ifft3, m)?)?;
    m.add_function(wrap_pyfunction!(demeaned_hann_power_spectrum_1d, m)?)?;
    Ok(())
}

