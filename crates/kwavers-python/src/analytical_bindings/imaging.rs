//! PyO3 bindings for `kwavers_physics::analytical::imaging`.

use kwavers_physics::analytical::imaging;
use kwavers_physics::analytical::pulse_echo::{
    bmode_db_fixed_reference as core_bmode_db_fixed_reference,
    delta_bmode_db as core_delta_bmode_db, simulate_receive_rf as core_simulate_receive_rf,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
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
    Ok(result.into_pyarray(py).unbind())
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
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Doppler frequency shift.
///
/// f_d = 2 * f0 * v * cos(theta) / c
///
/// Args:
///     v_m_s: Scatterer velocity [m/s].
///     theta_rad: Angle between beam and velocity vector [rad].
///     f0_hz: Transmit centre frequency [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Doppler shift [Hz].
#[pyfunction]
#[pyo3(signature = (v_m_s, theta_rad, f0_hz, c))]
pub fn doppler_frequency_shift(v_m_s: f64, theta_rad: f64, f0_hz: f64, c: f64) -> PyResult<f64> {
    Ok(imaging::doppler_frequency_shift(v_m_s, theta_rad, f0_hz, c))
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
    Ok(result.into_pyarray(py).unbind())
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

/// First-Born synthetic-aperture pulse-echo channel RF from point scatterers.
///
/// Each scatterer re-radiates a Gaussian-modulated tone burst reaching element `s`
/// at the one-way time of flight `|r_i − r_s|/c` (1/r spreading, reflectivity
/// weighting); contributions sum coherently. Beamforming the result with the one-way
/// `beamform_image_delay_and_sum` reconstructs the reflectivity map — a genuine
/// receive-data → B-mode pipeline.
///
/// Args:
///     scat_pos: (n_scat, 3) scatterer positions [m].
///     scat_amp: (n_scat,) reflectivity weights.
///     elem_pos: (n_elem, 3) array element positions [m].
///     c, fs, f0: sound speed [m/s], sampling [Hz], imaging centre frequency [Hz].
///     frac_bw: fractional −6 dB pulse bandwidth.
///     n_samples: RF record length [samples].
///
/// Returns:
///     (n_elem, n_samples) channel RF.
#[pyfunction]
#[pyo3(signature = (scat_pos, scat_amp, elem_pos, c, fs, f0, n_samples, frac_bw=0.6))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_receive_rf<'py>(
    py: Python<'py>,
    scat_pos: PyReadonlyArray2<'py, f64>,
    scat_amp: PyReadonlyArray1<'py, f64>,
    elem_pos: PyReadonlyArray2<'py, f64>,
    c: f64,
    fs: f64,
    f0: f64,
    n_samples: usize,
    frac_bw: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let sp = scat_pos.as_array();
    let sa = scat_amp.as_array();
    let ep = elem_pos.as_array();
    if sp.ncols() != 3 || ep.ncols() != 3 {
        return Err(PyValueError::new_err(
            "scat_pos and elem_pos must have shape (n, 3)",
        ));
    }
    if sp.nrows() != sa.len() {
        return Err(PyValueError::new_err(
            "scat_pos rows must match scat_amp length",
        ));
    }
    let rf = core_simulate_receive_rf(sp, sa, ep, c, fs, f0, frac_bw, n_samples);
    Ok(rf.into_pyarray(py).unbind())
}

/// Log-compress an envelope image with a fixed sequence reference.
#[pyfunction]
#[pyo3(signature = (envelope, reference, floor_db))]
pub fn bmode_db_fixed_reference(
    py: Python<'_>,
    envelope: PyReadonlyArray1<f64>,
    reference: f64,
    floor_db: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let env = envelope
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = py.detach(|| core_bmode_db_fixed_reference(env, reference, floor_db));
    Ok(out.into_pyarray(py).unbind())
}

/// Baseline-relative delta B-mode in dB.
#[pyfunction]
#[pyo3(signature = (envelope, baseline, epsilon=1.0e-12))]
pub fn delta_bmode_db(
    py: Python<'_>,
    envelope: PyReadonlyArray1<f64>,
    baseline: PyReadonlyArray1<f64>,
    epsilon: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let env = envelope
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let base = baseline
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = py.detach(|| core_delta_bmode_db(env, base, epsilon));
    Ok(out.into_pyarray(py).unbind())
}
