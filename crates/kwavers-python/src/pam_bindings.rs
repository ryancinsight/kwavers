//! Passive acoustic mapping bindings.
//!
//! These functions keep the Python boundary thin: NumPy arrays are borrowed as
//! read-only ndarray views and the authoritative validation/beamforming contract
//! lives in `kwavers::analysis::signal_processing::pam`.

use kwavers::analysis::signal_processing::beamforming::{
    beamform_image_das, ImagingDasApodization, ImagingDasConfig,
};
use kwavers::analysis::signal_processing::pam::delay_and_sum::ApodizationType;
use kwavers::analysis::signal_processing::pam::{DelayAndSumConfig, DelayAndSumPAM};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub fn register_pam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(passive_acoustic_map_das, m)?)?;
    m.add_function(wrap_pyfunction!(beamform_image_delay_and_sum, m)?)?;
    Ok(())
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
    let intensity = pam
        .beamform_view(passive_data.as_array(), grid_points.as_array())
        .map_err(|err| PyRuntimeError::new_err(format!("kwavers PAM error: {err}")))?;

    Ok(PyArray1::from_owned_array(py, intensity).into())
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

    let image = beamform_image_das(
        sensor_data.as_array(),
        sensor_positions.as_array(),
        grid_points.as_array(),
        &config,
    )
    .map_err(|err| PyRuntimeError::new_err(format!("kwavers imaging_das error: {err}")))?;

    Ok(PyArray1::from_owned_array(py, image).into())
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
