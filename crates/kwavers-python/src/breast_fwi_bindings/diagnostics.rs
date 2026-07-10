//! PyO3 wrappers for Ali 2025 breast-FWI diagnostic metrics.

use super::complex_compat::{leto3_to_nd3, nc_to_ec3, nd_to_leto3};
use super::{PyBreastFwiPstdDatasetConfig, PyMultiRowRingArray};
use kwavers_diagnostics::reconstruction::breast_ust_fwi::{
    acquisition_identifiability as breast_ust_acquisition_identifiability,
    diagnose_breast_ust_observation_pair,
    passive_receiver_mask as breast_ust_passive_receiver_mask,
    reconstruction_metrics as breast_ust_reconstruction_metrics,
    scaled_observation_residual_metrics as breast_ust_scaled_observation_residual_metrics,
    sine_frequency_bin_coefficient as breast_ust_sine_frequency_bin_coefficient,
    source_channel_residual_diagnostics as breast_ust_source_channel_residual_diagnostics,
    source_excitation_diagnostics as breast_ust_source_excitation_diagnostics,
    source_receiver_mask as breast_ust_source_receiver_mask,
    table1_parity as breast_ust_table1_parity, BreastUstAcquisitionIdentifiability,
    BreastUstObservationPairDiagnostics, BreastUstReconstructionMetrics,
    BreastUstScaledObservationResidualMetrics, BreastUstSourceChannelResidualDiagnostics,
    BreastUstSourceExcitationDiagnostics, BreastUstSourceExcitationFrequencyDiagnostics,
    BreastUstSourceScalingPolicy, BreastUstTable1Parity,
};
use eunomia::Complex64;
use numpy::{ToPyArray, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn diagnose_breast_fwi_observation_pair<'py>(
    py: Python<'py>,
    predicted_pressure: PyReadonlyArray3<'py, Complex64>,
    observed_pressure: PyReadonlyArray3<'py, Complex64>,
    array: &PyMultiRowRingArray,
    frequencies_hz: Vec<f64>,
    config: &PyBreastFwiPstdDatasetConfig,
    time_steps_per_frequency: Vec<usize>,
    frequency_bin_start_steps_per_frequency: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let predicted = nd_to_leto3(nc_to_ec3(predicted_pressure.as_array().to_owned()));
    let observed = nd_to_leto3(nc_to_ec3(observed_pressure.as_array().to_owned()));
    let diagnostics = py
        .detach(|| {
            diagnose_breast_ust_observation_pair(
                &predicted,
                &observed,
                &array.inner,
                &frequencies_hz,
                config.inner,
                &time_steps_per_frequency,
                &frequency_bin_start_steps_per_frequency,
            )
        })
        .map_err(kwavers_to_value_py)?;
    observation_pair_diagnostics_to_dict(py, &diagnostics)
}

#[pyfunction]
pub fn breast_fwi_scaled_observation_residual_metrics<'py>(
    py: Python<'py>,
    predicted_pressure: PyReadonlyArray3<'py, Complex64>,
    observed_pressure: PyReadonlyArray3<'py, Complex64>,
    receiver_mask: Option<PyReadonlyArray3<'py, bool>>,
) -> PyResult<Bound<'py, PyDict>> {
    let predicted = nd_to_leto3(nc_to_ec3(predicted_pressure.as_array().to_owned()));
    let observed = nd_to_leto3(nc_to_ec3(observed_pressure.as_array().to_owned()));
    let mask = receiver_mask.map(|mask| nd_to_leto3(mask.as_array().to_owned()));
    let metrics = py
        .detach(|| {
            breast_ust_scaled_observation_residual_metrics(&predicted, &observed, mask.as_ref())
        })
        .map_err(kwavers_to_value_py)?;
    scaled_residual_metrics_to_dict(py, &metrics)
}

#[pyfunction]
pub fn breast_fwi_source_channel_residual_diagnostics<'py>(
    py: Python<'py>,
    predicted_pressure: PyReadonlyArray3<'py, Complex64>,
    observed_pressure: PyReadonlyArray3<'py, Complex64>,
    circumferential_elements: usize,
    rows: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let predicted = nd_to_leto3(nc_to_ec3(predicted_pressure.as_array().to_owned()));
    let observed = nd_to_leto3(nc_to_ec3(observed_pressure.as_array().to_owned()));
    let diagnostics = py
        .detach(|| {
            breast_ust_source_channel_residual_diagnostics(
                &predicted,
                &observed,
                circumferential_elements,
                rows,
            )
        })
        .map_err(kwavers_to_value_py)?;
    source_channel_diagnostics_to_dict(py, &diagnostics)
}

#[pyfunction]
pub fn breast_fwi_source_receiver_mask<'py>(
    py: Python<'py>,
    observation_shape: (usize, usize, usize),
    circumferential_elements: usize,
    rows: usize,
) -> PyResult<Py<PyArray3<bool>>> {
    let mask = breast_ust_source_receiver_mask(observation_shape, circumferential_elements, rows)
        .map_err(kwavers_to_value_py)?;
    Ok(leto3_to_nd3(mask).to_pyarray(py).into())
}

#[pyfunction]
pub fn breast_fwi_passive_receiver_mask<'py>(
    py: Python<'py>,
    observation_shape: (usize, usize, usize),
    circumferential_elements: usize,
    rows: usize,
) -> PyResult<Py<PyArray3<bool>>> {
    let mask = breast_ust_passive_receiver_mask(observation_shape, circumferential_elements, rows)
        .map_err(kwavers_to_value_py)?;
    Ok(leto3_to_nd3(mask).to_pyarray(py).into())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn breast_fwi_source_excitation_diagnostics<'py>(
    py: Python<'py>,
    predicted_pressure: PyReadonlyArray3<'py, Complex64>,
    observed_pressure: PyReadonlyArray3<'py, Complex64>,
    frequencies_hz: Vec<f64>,
    source_amplitude_pa: f64,
    time_step_s: f64,
    time_steps_per_frequency: Vec<usize>,
    frequency_bin_start_steps_per_frequency: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let predicted = nd_to_leto3(nc_to_ec3(predicted_pressure.as_array().to_owned()));
    let observed = nd_to_leto3(nc_to_ec3(observed_pressure.as_array().to_owned()));
    let diagnostics = py
        .detach(|| {
            breast_ust_source_excitation_diagnostics(
                &predicted,
                &observed,
                &frequencies_hz,
                source_amplitude_pa,
                time_step_s,
                &time_steps_per_frequency,
                &frequency_bin_start_steps_per_frequency,
            )
        })
        .map_err(kwavers_to_value_py)?;
    source_excitation_diagnostics_to_dict(py, &diagnostics)
}

#[pyfunction]
pub fn breast_fwi_sine_frequency_bin_coefficient(
    frequency_hz: f64,
    time_step_s: f64,
    total_steps: usize,
    start_sample: usize,
) -> PyResult<(f64, f64)> {
    let coefficient = breast_ust_sine_frequency_bin_coefficient(
        frequency_hz,
        time_step_s,
        total_steps,
        start_sample,
    )
    .map_err(kwavers_to_value_py)?;
    Ok((coefficient.re, coefficient.im))
}

#[pyfunction]
pub fn breast_fwi_acquisition_identifiability<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize),
    frequencies_hz: Vec<f64>,
    transmissions: usize,
    receivers: usize,
    source_scaling_policy: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let policy =
        BreastUstSourceScalingPolicy::parse(source_scaling_policy).map_err(kwavers_to_value_py)?;
    let report = breast_ust_acquisition_identifiability(
        shape,
        &frequencies_hz,
        transmissions,
        receivers,
        policy,
    )
    .map_err(kwavers_to_value_py)?;
    acquisition_identifiability_to_dict(py, &report)
}

#[pyfunction]
pub fn breast_fwi_reconstruction_metrics<'py>(
    py: Python<'py>,
    reference_m_s: PyReadonlyArray3<'py, f64>,
    estimate_m_s: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let reference = nd_to_leto3(reference_m_s.as_array().to_owned());
    let estimate = nd_to_leto3(estimate_m_s.as_array().to_owned());
    let metrics = py
        .detach(|| breast_ust_reconstruction_metrics(&reference, &estimate))
        .map_err(kwavers_to_value_py)?;
    reconstruction_metrics_to_dict(py, &metrics)
}

#[pyfunction]
pub fn breast_fwi_table1_parity<'py>(
    py: Python<'py>,
    rmse_m_s: f64,
    pearson_correlation: f64,
    phantom_index: usize,
    rmse_multiplier: f64,
    pcc_fraction: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let parity = breast_ust_table1_parity(
        rmse_m_s,
        pearson_correlation,
        phantom_index,
        rmse_multiplier,
        pcc_fraction,
    )
    .map_err(kwavers_to_value_py)?;
    table1_parity_to_dict(py, &parity)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(diagnose_breast_fwi_observation_pair, m)?)?;
    m.add_function(wrap_pyfunction!(
        breast_fwi_scaled_observation_residual_metrics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        breast_fwi_source_channel_residual_diagnostics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(breast_fwi_source_receiver_mask, m)?)?;
    m.add_function(wrap_pyfunction!(breast_fwi_passive_receiver_mask, m)?)?;
    m.add_function(wrap_pyfunction!(
        breast_fwi_source_excitation_diagnostics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        breast_fwi_sine_frequency_bin_coefficient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(breast_fwi_acquisition_identifiability, m)?)?;
    m.add_function(wrap_pyfunction!(breast_fwi_reconstruction_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(breast_fwi_table1_parity, m)?)?;
    Ok(())
}

fn observation_pair_diagnostics_to_dict<'py>(
    py: Python<'py>,
    diagnostics: &BreastUstObservationPairDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "forward_consistency",
        scaled_residual_metrics_to_dict(py, &diagnostics.forward_consistency)?,
    )?;
    out.set_item(
        "source_channel_consistency",
        source_channel_diagnostics_to_dict(py, &diagnostics.source_channel_consistency)?,
    )?;
    out.set_item(
        "source_excitation",
        source_excitation_diagnostics_to_dict(py, &diagnostics.source_excitation)?,
    )?;
    Ok(out)
}

fn scaled_residual_metrics_to_dict<'py>(
    py: Python<'py>,
    metrics: &BreastUstScaledObservationResidualMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("frequency_count", metrics.frequency_count)?;
    out.set_item("transmission_count", metrics.transmission_count)?;
    out.set_item("receiver_count", metrics.receiver_count)?;
    out.set_item("row_count", metrics.row_count)?;
    out.set_item("selected_receiver_count", metrics.selected_receiver_count)?;
    out.set_item("observed_l2_norm", metrics.observed_l2_norm)?;
    out.set_item("scaled_residual_l2_norm", metrics.scaled_residual_l2_norm)?;
    out.set_item("normalized_l2_residual", metrics.normalized_l2_residual)?;
    out.set_item("max_abs_scaled_residual", metrics.max_abs_scaled_residual)?;
    out.set_item(
        "row_normalized_l2_residual_mean",
        metrics.row_normalized_l2_residual_mean,
    )?;
    out.set_item(
        "row_normalized_l2_residual_max",
        metrics.row_normalized_l2_residual_max,
    )?;
    out.set_item(
        "source_scale_magnitude_min",
        metrics.source_scale_magnitude_min,
    )?;
    out.set_item(
        "source_scale_magnitude_max",
        metrics.source_scale_magnitude_max,
    )?;
    Ok(out)
}

fn source_channel_diagnostics_to_dict<'py>(
    py: Python<'py>,
    diagnostics: &BreastUstSourceChannelResidualDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "active_receiver_count_per_row",
        diagnostics.active_receiver_count_per_row,
    )?;
    out.set_item(
        "passive_receiver_count_per_row",
        diagnostics.passive_receiver_count_per_row,
    )?;
    out.set_item(
        "all_channel_normalized_l2_residual",
        diagnostics.all_channel_normalized_l2_residual,
    )?;
    out.set_item(
        "passive_only_normalized_l2_residual",
        diagnostics.passive_only_normalized_l2_residual,
    )?;
    out.set_item(
        "passive_only_scaled_residual_l2_norm",
        diagnostics.passive_only_scaled_residual_l2_norm,
    )?;
    out.set_item(
        "active_full_scale_residual_l2_norm",
        diagnostics.active_full_scale_residual_l2_norm,
    )?;
    out.set_item(
        "passive_full_scale_residual_l2_norm",
        diagnostics.passive_full_scale_residual_l2_norm,
    )?;
    out.set_item(
        "active_full_scale_observed_l2_norm",
        diagnostics.active_full_scale_observed_l2_norm,
    )?;
    out.set_item(
        "passive_full_scale_observed_l2_norm",
        diagnostics.passive_full_scale_observed_l2_norm,
    )?;
    out.set_item(
        "active_full_scale_normalized_l2_residual",
        diagnostics.active_full_scale_normalized_l2_residual,
    )?;
    out.set_item(
        "passive_full_scale_normalized_l2_residual",
        diagnostics.passive_full_scale_normalized_l2_residual,
    )?;
    out.set_item(
        "active_full_scale_residual_energy_fraction",
        diagnostics.active_full_scale_residual_energy_fraction,
    )?;
    out.set_item(
        "passive_full_scale_residual_energy_fraction",
        diagnostics.passive_full_scale_residual_energy_fraction,
    )?;
    Ok(out)
}

fn source_excitation_diagnostics_to_dict<'py>(
    py: Python<'py>,
    diagnostics: &BreastUstSourceExcitationDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("frequency_count", diagnostics.frequency_count)?;
    out.set_item("transmission_count", diagnostics.transmission_count)?;
    out.set_item("source_amplitude_pa", diagnostics.source_amplitude_pa)?;
    out.set_item(
        "max_source_scale_magnitude_coefficient_of_variation",
        diagnostics.max_source_scale_magnitude_coefficient_of_variation,
    )?;
    out.set_item(
        "max_source_scale_phase_circular_variance",
        diagnostics.max_source_scale_phase_circular_variance,
    )?;
    out.set_item(
        "max_source_scale_phase_span_rad",
        diagnostics.max_source_scale_phase_span_rad,
    )?;
    let rows = PyList::empty(py);
    for row in &diagnostics.per_frequency {
        rows.append(source_excitation_frequency_to_dict(py, row)?)?;
    }
    out.set_item("per_frequency", rows)?;
    Ok(out)
}

fn source_excitation_frequency_to_dict<'py>(
    py: Python<'py>,
    row: &BreastUstSourceExcitationFrequencyDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("frequency_hz", row.frequency_hz)?;
    out.set_item("tone_bin_magnitude", row.tone_bin_magnitude)?;
    out.set_item("tone_bin_phase_rad", row.tone_bin_phase_rad)?;
    out.set_item(
        "mean_source_scale_magnitude",
        row.mean_source_scale_magnitude,
    )?;
    out.set_item(
        "source_scale_magnitude_coefficient_of_variation",
        row.source_scale_magnitude_coefficient_of_variation,
    )?;
    out.set_item(
        "source_scale_phase_circular_variance",
        row.source_scale_phase_circular_variance,
    )?;
    out.set_item(
        "source_scale_phase_span_rad",
        row.source_scale_phase_span_rad,
    )?;
    Ok(out)
}

fn acquisition_identifiability_to_dict<'py>(
    py: Python<'py>,
    report: &BreastUstAcquisitionIdentifiability,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("unknown_voxels", report.unknown_voxels)?;
    out.set_item("frequency_count", report.frequency_count)?;
    out.set_item("complex_observations", report.complex_observations)?;
    out.set_item("real_observation_dof", report.real_observation_dof)?;
    out.set_item(
        "source_scaling_policy",
        report.source_scaling_policy.as_str(),
    )?;
    out.set_item(
        "estimated_source_scale_real_dof",
        report.estimated_source_scale_real_dof,
    )?;
    out.set_item(
        "informative_real_dof_upper_bound",
        report.informative_real_dof_upper_bound,
    )?;
    out.set_item(
        "informative_dof_to_unknown_ratio",
        report.informative_dof_to_unknown_ratio,
    )?;
    out.set_item(
        "underdetermined_by_rank_upper_bound",
        report.underdetermined_by_rank_upper_bound,
    )?;
    Ok(out)
}

fn reconstruction_metrics_to_dict<'py>(
    py: Python<'py>,
    metrics: &BreastUstReconstructionMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("rmse_m_s", metrics.rmse_m_s)?;
    out.set_item("normalized_rmse", metrics.normalized_rmse)?;
    out.set_item("pearson_correlation", metrics.pearson_correlation)?;
    out.set_item("reference_min_m_s", metrics.reference_min_m_s)?;
    out.set_item("reference_max_m_s", metrics.reference_max_m_s)?;
    out.set_item("estimate_min_m_s", metrics.estimate_min_m_s)?;
    out.set_item("estimate_max_m_s", metrics.estimate_max_m_s)?;
    Ok(out)
}

fn table1_parity_to_dict<'py>(
    py: Python<'py>,
    parity: &BreastUstTable1Parity,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("phantom_index", parity.phantom_index)?;
    out.set_item("table1_3d_rmse_m_s", parity.table1_3d_rmse_m_s)?;
    out.set_item(
        "table1_3d_pearson_correlation",
        parity.table1_3d_pearson_correlation,
    )?;
    out.set_item("rmse_threshold_m_s", parity.rmse_threshold_m_s)?;
    out.set_item("pcc_threshold", parity.pcc_threshold)?;
    out.set_item("rmse_pass", parity.rmse_pass)?;
    out.set_item("pcc_pass", parity.pcc_pass)?;
    out.set_item("passes", parity.passes)?;
    Ok(out)
}

fn kwavers_to_value_py(err: kwavers_core::error::KwaversError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

