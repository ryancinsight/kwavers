//! Passive acoustic mapping bindings.
//!
//! These functions keep the Python boundary thin: NumPy arrays are borrowed as
//! read-only ndarray views and the authoritative validation/beamforming contract
//! lives in `kwavers_analysis::signal_processing::pam`.

use crate::breast_fwi_bindings::complex_compat::{leto1_to_nd1, nd_to_leto2};
use kwavers_analysis::signal_processing::beamforming::adaptive::subspace::MUSIC;
use kwavers_analysis::signal_processing::beamforming::{
    beamform_image_das, ImagingDasApodization, ImagingDasConfig,
};
use kwavers_analysis::signal_processing::pam::delay_and_sum::ApodizationType;
use kwavers_analysis::signal_processing::pam::{
    eigenspace_covariance_eigenvalues as compute_eigenspace_covariance_eigenvalues,
    DelayAndSumConfig, DelayAndSumPAM,
};
use kwavers_math::fft::Complex64 as KwComplex;
use kwavers_math::linear_algebra::eigendecomposition::{EigenSolver, EigenSolverConfig};
use leto::{Array1, Array2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub fn register_pam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(passive_acoustic_map_das, m)?)?;
    m.add_function(wrap_pyfunction!(beamform_image_delay_and_sum, m)?)?;
    m.add_function(wrap_pyfunction!(hermitian_eigenvalues_complex, m)?)?;
    m.add_function(wrap_pyfunction!(eigenspace_covariance_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(music_pseudospectrum, m)?)?;
    Ok(())
}

/// Build a Hermitian `Array2<Complex64>` from real/imag parts, symmetrised to
/// `½(R + Rᴴ)` to absorb float round-off so the strict Hermitian check passes.
fn hermitian_from_parts(
    re: &PyReadonlyArray2<f64>,
    im: &PyReadonlyArray2<f64>,
) -> PyResult<Array2<KwComplex>> {
    let re = re.as_array();
    let im = im.as_array();
    let n = re.nrows();
    if re.ncols() != n || im.dim() != (n, n) {
        return Err(PyValueError::new_err(
            "covariance real/imag parts must be the same square (N×N) shape",
        ));
    }
    let mut r = Array2::<KwComplex>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            r[[i, j]] = KwComplex::new(re[[i, j]], im[[i, j]]);
        }
    }
    let mut h = Array2::<KwComplex>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            h[[i, j]] = KwComplex::new(
                0.5 * (r[[i, j]].re + r[[j, i]].re),
                0.5 * (r[[i, j]].im - r[[j, i]].im),
            );
        }
    }
    Ok(h)
}

/// Eigenvalues of a Hermitian covariance / cross-spectral matrix, sorted
/// descending — the signal/noise eigenvalue split underlying MUSIC and
/// eigenspace-MV (book §22.4). The matrix is passed as separate real and
/// imaginary `N×N` parts and symmetrised before the eigendecomposition.
///
/// Returns the `N` real eigenvalues (descending).
#[pyfunction]
#[pyo3(signature = (covariance_real, covariance_imag))]
fn hermitian_eigenvalues_complex(
    py: Python<'_>,
    covariance_real: PyReadonlyArray2<f64>,
    covariance_imag: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let h = hermitian_from_parts(&covariance_real, &covariance_imag)?;
    let result = EigenSolver::jacobi_hermitian(&h, EigenSolverConfig::default())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut v = result.eigenvalues.into_vec();
    v.sort_by(|a, b| b.total_cmp(a));
    let eigenvalues = Array1::from(v);
    Ok(PyArray1::from_owned_array(py, leto1_to_nd1(eigenvalues)).into())
}

/// Deterministic Theorem 22.2 eigenspace PAM covariance eigenvalues.
///
/// Returns `n_sources` values equal to `signal_power + noise_power`, followed
/// by `n_elements - n_sources` values equal to `noise_power`.
#[pyfunction]
#[pyo3(signature = (n_elements, n_sources, signal_power, noise_power))]
fn eigenspace_covariance_eigenvalues(
    py: Python<'_>,
    n_elements: usize,
    n_sources: usize,
    signal_power: f64,
    noise_power: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let eigenvalues =
        compute_eigenspace_covariance_eigenvalues(n_elements, n_sources, signal_power, noise_power)
            .map_err(|err| PyValueError::new_err(format!("kwavers eigenspace PAM error: {err}")))?;

    Ok(PyArray1::from_vec(py, eigenvalues).into())
}

/// MUSIC noise-subspace pseudospectrum `P(θ) = 1/‖E_nᴴ a(θ)‖²` for one steering
/// vector against a Hermitian covariance matrix (Schmidt 1986; book §22.4).
///
/// `covariance_{real,imag}` are the `N×N` parts; `steering_{real,imag}` are the
/// length-`N` steering-vector parts; `num_sources` is the signal-subspace
/// dimension `K` (must satisfy `0 < K < N`).
#[pyfunction]
#[pyo3(signature = (covariance_real, covariance_imag, steering_real, steering_imag, num_sources))]
fn music_pseudospectrum(
    covariance_real: PyReadonlyArray2<f64>,
    covariance_imag: PyReadonlyArray2<f64>,
    steering_real: PyReadonlyArray1<f64>,
    steering_imag: PyReadonlyArray1<f64>,
    num_sources: usize,
) -> PyResult<f64> {
    let h = hermitian_from_parts(&covariance_real, &covariance_imag)?;
    let sr = steering_real
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let si = steering_imag
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if sr.len() != h.shape()[0] || si.len() != h.shape()[0] {
        return Err(PyValueError::new_err(
            "steering real/imag parts must have length N matching the covariance",
        ));
    }
    let steering = Array1::from_iter(
        sr.iter()
            .zip(si.iter())
            .map(|(&a, &b)| KwComplex::new(a, b)),
    );
    MUSIC::new(num_sources)
        .pseudospectrum(&h, &steering)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    passive_data,
    sensor_positions,
    grid_points,
    sound_speed,
    sampling_frequency,
    *,
    window_size = 512,
    apodization = "hamming",
    coherence_weighting = true
))]
#[allow(clippy::too_many_arguments)]
fn passive_acoustic_map_das<'py>(
    py: Python<'py>,
    passive_data: PyReadonlyArray2<f64>,
    sensor_positions: PyReadonlyArray2<f64>,
    grid_points: PyReadonlyArray2<f64>,
    sound_speed: f64,
    sampling_frequency: f64,
    window_size: usize,
    apodization: &str,
    coherence_weighting: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let sensor_positions = sensor_positions.as_array();
    if sensor_positions.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "sensor_positions must have shape [sensors x 3], got {} columns",
            sensor_positions.ncols()
        )));
    }
    if !sensor_positions.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "sensor_positions must contain only finite coordinates",
        ));
    }

    let sensors: Vec<[f64; 3]> = sensor_positions
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let config = DelayAndSumConfig {
        sound_speed,
        sampling_frequency,
        window_size,
        apodization: parse_apodization(apodization)?,
        coherence_weighting,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config)
        .map_err(|err| PyValueError::new_err(format!("kwavers PAM config error: {err}")))?;
    let passive_leto = nd_to_leto2(passive_data.as_array().to_owned());
    let grid_leto = nd_to_leto2(grid_points.as_array().to_owned());
    let intensity = pam
        .beamform_view(passive_leto.view(), grid_leto.view())
        .map_err(|err| PyRuntimeError::new_err(format!("kwavers PAM error: {err}")))?;

    Ok(PyArray1::from_owned_array(py, leto1_to_nd1(intensity)).into())
}

/// Active-imaging delay-and-sum reconstruction.
///
/// Reconstructs a pressure-like image by computing one-way TOF from each grid
/// point to each sensor and coherently summing apodization-weighted, linearly
/// interpolated samples. Mirrors `KWave.jl beamform_delay_and_sum` and is
/// distinct from `passive_acoustic_map_das` (energy/intensity map).
#[pyfunction]
#[pyo3(signature = (
    sensor_data,
    sensor_positions,
    grid_points,
    sound_speed,
    sampling_frequency,
    *,
    apodization = "rectangular"
))]
fn beamform_image_delay_and_sum<'py>(
    py: Python<'py>,
    sensor_data: PyReadonlyArray2<f64>,
    sensor_positions: PyReadonlyArray2<f64>,
    grid_points: PyReadonlyArray2<f64>,
    sound_speed: f64,
    sampling_frequency: f64,
    apodization: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let config = ImagingDasConfig::new(
        sound_speed,
        sampling_frequency,
        parse_imaging_apodization(apodization)?,
    )
    .map_err(|err| PyValueError::new_err(format!("kwavers imaging_das config error: {err}")))?;

    let sensor_leto = nd_to_leto2(sensor_data.as_array().to_owned());
    let positions_leto = nd_to_leto2(sensor_positions.as_array().to_owned());
    let grid_leto = nd_to_leto2(grid_points.as_array().to_owned());
    let image = beamform_image_das(
        sensor_leto.view(),
        positions_leto.view(),
        grid_leto.view(),
        &config,
    )
    .map_err(|err| PyRuntimeError::new_err(format!("kwavers imaging_das error: {err}")))?;

    Ok(PyArray1::from_owned_array(py, leto1_to_nd1(image)).into())
}

fn parse_imaging_apodization(value: &str) -> PyResult<ImagingDasApodization> {
    match value.to_ascii_lowercase().as_str() {
        "none" | "rectangular" | "boxcar" => Ok(ImagingDasApodization::Rectangular),
        "hamming" => Ok(ImagingDasApodization::Hamming),
        "hann" | "hanning" => Ok(ImagingDasApodization::Hanning),
        "blackman" => Ok(ImagingDasApodization::Blackman),
        other => Err(PyValueError::new_err(format!(
            "apodization must be one of none, rectangular, hamming, hann, blackman; got {other}"
        ))),
    }
}

fn parse_apodization(value: &str) -> PyResult<ApodizationType> {
    match value.to_ascii_lowercase().as_str() {
        "none" | "rectangular" | "boxcar" => Ok(ApodizationType::Uniform),
        "hamming" => Ok(ApodizationType::Hamming),
        "hann" | "hanning" => Ok(ApodizationType::Hanning),
        "blackman" => Ok(ApodizationType::Blackman),
        other => Err(PyValueError::new_err(format!(
            "apodization must be one of none, hamming, hann, hanning, blackman; got {other}"
        ))),
    }
}
