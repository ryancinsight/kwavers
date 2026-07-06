use kwavers_diagnostics::reconstruction::transcranial_ust::{
    reconstruct_brain_slice, resample_head_slice, select_head_slice, AcousticSlice,
    TranscranialUstBornInversionConfig,
};
use kwavers_solver::inverse::linear_born_inversion::LinearBornInversionConfig;
use ndarray::Array1;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use crate::ritk_image::load_ritk_nifti;

use super::kwavers_to_py;

/// Run encoded 1024-element transcranial UST Born inversion from a RITK-loaded CT.
#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    grid_size = 72,
    slice_index = None,
    iterations = 24,
    radius_m = None,
    frequencies_hz = None,
    receiver_offsets = None,
    frequency_continuation = true,
    sobolev_radius_voxels = 1,
    sobolev_weight = 0.35,
    enhancement_gain = 0.65,
    edge_preserving_weight = 0.0001,
    edge_preserving_epsilon = 0.004,
    edge_preserving_step = 0.12,
    edge_preserving_iterations = 1,
    attenuation_model = true,
    nonlinear_harmonic_model = true,
    source_pressure_mpa = 0.15,
    nonlinear_beta = 4.5
))]
#[allow(clippy::too_many_arguments)]
pub fn run_transcranial_ust_slice_inversion_from_ritk_ct<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    grid_size: usize,
    slice_index: Option<usize>,
    iterations: usize,
    radius_m: Option<f64>,
    frequencies_hz: Option<Vec<f64>>,
    receiver_offsets: Option<Vec<usize>>,
    frequency_continuation: bool,
    sobolev_radius_voxels: usize,
    sobolev_weight: f64,
    enhancement_gain: f64,
    edge_preserving_weight: f64,
    edge_preserving_epsilon: f64,
    edge_preserving_step: f64,
    edge_preserving_iterations: usize,
    attenuation_model: bool,
    nonlinear_harmonic_model: bool,
    source_pressure_mpa: f64,
    nonlinear_beta: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let path = Path::new(ct_nifti_path);
    let (mut volume_hu, spacing_mm) = load_ritk_nifti(path)?;
    volume_hu.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
    let selected_slice = match slice_index {
        Some(idx) => idx,
        None => select_head_slice(&volume_hu).map_err(kwavers_to_py)?,
    };
    let resampled = resample_head_slice(&volume_hu, spacing_mm, selected_slice, grid_size)
        .map_err(kwavers_to_py)?;
    let acoustic = AcousticSlice::from_ct_hu_at_offset(
        resampled.hu.clone(),
        resampled.spacing_m,
        resampled.slice_offset_m,
    )
    .map_err(kwavers_to_py)?;

    let mut config = TranscranialUstBornInversionConfig {
        linear: LinearBornInversionConfig {
            iterations,
            ..LinearBornInversionConfig::default()
        },
        ..TranscranialUstBornInversionConfig::default()
    };
    if let Some(radius) = radius_m {
        config.radius_m = radius;
    }
    if let Some(freqs) = frequencies_hz {
        config.linear.frequencies_hz = freqs;
    }
    if let Some(offsets) = receiver_offsets {
        config.linear.receiver_offsets = offsets;
    }
    config.linear.frequency_continuation = frequency_continuation;
    config.linear.sobolev_radius_voxels = sobolev_radius_voxels;
    config.linear.sobolev_weight = sobolev_weight;
    config.linear.enhancement_gain = enhancement_gain;
    config.linear.edge_preserving_weight = edge_preserving_weight;
    config.linear.edge_preserving_epsilon = edge_preserving_epsilon;
    config.linear.edge_preserving_step = edge_preserving_step;
    config.linear.edge_preserving_iterations = edge_preserving_iterations;
    config.linear.attenuation_model = attenuation_model;
    config.linear.nonlinear_harmonic_model = nonlinear_harmonic_model;
    config.linear.source_pressure_mpa = source_pressure_mpa;
    config.linear.nonlinear_beta = nonlinear_beta;

    let result = py
        .detach(|| reconstruct_brain_slice(&acoustic, &config))
        .map_err(kwavers_to_py)?;
    let harmonic_count = config.harmonic_count();

    let out = PyDict::new(py);
    let metrics = PyDict::new(py);
    metrics.set_item("initial_objective", result.metrics.initial_objective)?;
    metrics.set_item("final_objective", result.metrics.final_objective)?;
    metrics.set_item(
        "objective_reduction_fraction",
        result.metrics.objective_reduction_fraction,
    )?;
    metrics.set_item(
        "migration_pearson_correlation",
        result.metrics.migration_pearson_correlation,
    )?;
    metrics.set_item(
        "migration_normalized_rmse",
        result.metrics.migration_normalized_rmse,
    )?;
    metrics.set_item(
        "migration_dynamic_range_m_s",
        result.metrics.migration_dynamic_range_m_s,
    )?;
    metrics.set_item("pearson_correlation", result.metrics.pearson_correlation)?;
    metrics.set_item("normalized_rmse", result.metrics.normalized_rmse)?;
    metrics.set_item(
        "enhanced_dynamic_range_m_s",
        result.metrics.enhanced_dynamic_range_m_s,
    )?;
    metrics.set_item("active_voxels", result.metrics.active_voxels)?;
    metrics.set_item("measurements", result.metrics.measurements)?;
    metrics.set_item("continuation_stages", result.metrics.continuation_stages)?;
    metrics.set_item(
        "target_dynamic_range_m_s",
        result.metrics.target_dynamic_range_m_s,
    )?;
    metrics.set_item(
        "reconstruction_dynamic_range_m_s",
        result.metrics.reconstruction_dynamic_range_m_s,
    )?;

    out.set_item("ct_hu", result.ct_hu.to_pyarray(py))?;
    out.set_item(
        "target_sound_speed_m_s",
        result.target_sound_speed_m_s.to_pyarray(py),
    )?;
    out.set_item(
        "initial_sound_speed_m_s",
        result.initial_sound_speed_m_s.to_pyarray(py),
    )?;
    out.set_item(
        "migration_sound_speed_m_s",
        result.migration_sound_speed_m_s.to_pyarray(py),
    )?;
    out.set_item(
        "reconstruction_sound_speed_m_s",
        result.reconstruction_sound_speed_m_s.to_pyarray(py),
    )?;
    out.set_item(
        "enhanced_reconstruction_sound_speed_m_s",
        result
            .enhanced_reconstruction_sound_speed_m_s
            .to_pyarray(py),
    )?;
    out.set_item("brain_mask", result.brain_mask.to_pyarray(py))?;
    out.set_item("skull_mask", result.skull_mask.to_pyarray(py))?;
    out.set_item(
        "synthetic_data",
        Array1::from(result.synthetic_data).to_pyarray(py),
    )?;
    out.set_item(
        "residual_history",
        Array1::from(result.residual_history).to_pyarray(py),
    )?;
    out.set_item("metrics", metrics)?;
    out.set_item("spacing_m", resampled.spacing_m)?;
    out.set_item("slice_offset_m", resampled.slice_offset_m)?;
    out.set_item("source_slice_index", resampled.source_slice_index)?;
    out.set_item("geometry_model", "hemispherical_cap")?;
    out.set_item("element_count", config.element_count)?;
    out.set_item("radius_m", config.radius_m)?;
    out.set_item("frequencies_hz", config.linear.frequencies_hz.clone())?;
    out.set_item("receiver_offsets", config.linear.receiver_offsets.clone())?;
    out.set_item(
        "frequency_continuation",
        config.linear.frequency_continuation,
    )?;
    out.set_item("sobolev_radius_voxels", config.linear.sobolev_radius_voxels)?;
    out.set_item("sobolev_weight", config.linear.sobolev_weight)?;
    out.set_item("enhancement_gain", config.linear.enhancement_gain)?;
    out.set_item(
        "edge_preserving_weight",
        config.linear.edge_preserving_weight,
    )?;
    out.set_item(
        "edge_preserving_epsilon",
        config.linear.edge_preserving_epsilon,
    )?;
    out.set_item("edge_preserving_step", config.linear.edge_preserving_step)?;
    out.set_item(
        "edge_preserving_iterations",
        config.linear.edge_preserving_iterations,
    )?;
    out.set_item("attenuation_model", config.linear.attenuation_model)?;
    out.set_item(
        "nonlinear_harmonic_model",
        config.linear.nonlinear_harmonic_model,
    )?;
    out.set_item("source_pressure_mpa", config.linear.source_pressure_mpa)?;
    out.set_item("nonlinear_beta", config.linear.nonlinear_beta)?;
    out.set_item("harmonic_count", harmonic_count)?;
    Ok(out)
}

