//! PyO3 wrapper for Ali 2025 forward-operator equivalence diagnostics.

use super::complex_compat::{nc_to_ec3, nd_to_leto3};
use kwavers_diagnostics::reconstruction::breast_ust_fwi::{
    forward_operator_equivalence_diagnostics_with_receiver_policy as breast_ust_forward_operator_equivalence_diagnostics_with_receiver_policy,
    scattering_increment_diagnostics as breast_ust_scattering_increment_diagnostics,
    BreastUstForwardOperatorEquivalenceDiagnostics, BreastUstForwardOperatorModelDiagnostics,
    BreastUstForwardOperatorPrediction, BreastUstReceiverChannelPolicy,
    BreastUstScatteringIncrementDiagnostics, BreastUstScatteringIncrementModelDiagnostics,
};
use leto::Array3;
use eunomia::Complex64;
use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

#[pyfunction]
#[pyo3(signature = (
    predictions_by_model,
    observed_pressure,
    frequencies_hz,
    source_amplitude_pa,
    time_step_s,
    time_steps_per_frequency,
    frequency_bin_start_steps_per_frequency,
    receiver_channel_policy = "all"
))]
#[allow(clippy::too_many_arguments)]
pub fn breast_fwi_operator_equivalence_diagnostics<'py>(
    py: Python<'py>,
    predictions_by_model: &Bound<'py, PyDict>,
    observed_pressure: PyReadonlyArray3<'py, Complex64>,
    frequencies_hz: Vec<f64>,
    source_amplitude_pa: f64,
    time_step_s: f64,
    time_steps_per_frequency: Vec<usize>,
    frequency_bin_start_steps_per_frequency: Vec<usize>,
    receiver_channel_policy: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let observed = nd_to_leto3(nc_to_ec3(observed_pressure.as_array().to_owned()));
    let receiver_channel_policy = BreastUstReceiverChannelPolicy::parse(receiver_channel_policy)
        .map_err(kwavers_to_value_py)?;
    let mut owned_predictions = Vec::<(String, Array3<_>)>::with_capacity(predictions_by_model.len());
    for (model, pressure) in predictions_by_model.iter() {
        let model = model.extract::<String>()?;
        let pressure = pressure.extract::<PyReadonlyArray3<'_, Complex64>>()?;
        owned_predictions.push((model, nd_to_leto3(nc_to_ec3(pressure.as_array().to_owned()))));
    }

    let diagnostics = py
        .detach(|| {
            let predictions = owned_predictions
                .iter()
                .map(|(model, pressure)| BreastUstForwardOperatorPrediction {
                    model: model.as_str(),
                    pressure,
                })
                .collect::<Vec<_>>();
            breast_ust_forward_operator_equivalence_diagnostics_with_receiver_policy(
                &predictions,
                &observed,
                &frequencies_hz,
                source_amplitude_pa,
                time_step_s,
                &time_steps_per_frequency,
                &frequency_bin_start_steps_per_frequency,
                receiver_channel_policy,
            )
        })
        .map_err(kwavers_to_value_py)?;
    operator_equivalence_to_dict(py, &diagnostics)
}

#[pyfunction]
#[pyo3(signature = (
    homogeneous_baseline,
    predictions_by_model,
    observed_pressure,
    receiver_channel_policy = "all"
))]
pub fn breast_fwi_scattering_increment_diagnostics<'py>(
    py: Python<'py>,
    homogeneous_baseline: PyReadonlyArray3<'py, Complex64>,
    predictions_by_model: &Bound<'py, PyDict>,
    observed_pressure: PyReadonlyArray3<'py, Complex64>,
    receiver_channel_policy: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let baseline = nd_to_leto3(nc_to_ec3(homogeneous_baseline.as_array().to_owned()));
    let observed = nd_to_leto3(nc_to_ec3(observed_pressure.as_array().to_owned()));
    let receiver_channel_policy = BreastUstReceiverChannelPolicy::parse(receiver_channel_policy)
        .map_err(kwavers_to_value_py)?;
    let mut owned_predictions = Vec::<(String, Array3<_>)>::with_capacity(predictions_by_model.len());
    for (model, pressure) in predictions_by_model.iter() {
        let model = model.extract::<String>()?;
        let pressure = pressure.extract::<PyReadonlyArray3<'_, Complex64>>()?;
        owned_predictions.push((model, nd_to_leto3(nc_to_ec3(pressure.as_array().to_owned()))));
    }

    let diagnostics = py
        .detach(|| {
            let predictions = owned_predictions
                .iter()
                .map(|(model, pressure)| BreastUstForwardOperatorPrediction {
                    model: model.as_str(),
                    pressure,
                })
                .collect::<Vec<_>>();
            breast_ust_scattering_increment_diagnostics(
                &baseline,
                &predictions,
                &observed,
                receiver_channel_policy,
            )
        })
        .map_err(kwavers_to_value_py)?;
    scattering_increment_to_dict(py, &diagnostics)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        breast_fwi_operator_equivalence_diagnostics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        breast_fwi_scattering_increment_diagnostics,
        m
    )?)?;
    Ok(())
}

fn operator_equivalence_to_dict<'py>(
    py: Python<'py>,
    diagnostics: &BreastUstForwardOperatorEquivalenceDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("model_count", diagnostics.model_count)?;
    out.set_item(
        "receiver_channel_policy",
        diagnostics.receiver_channel_policy.as_str(),
    )?;
    out.set_item("best_model", diagnostics.best_model.as_str())?;
    out.set_item(
        "best_normalized_l2_residual",
        diagnostics.best_normalized_l2_residual,
    )?;
    out.set_item("worst_model", diagnostics.worst_model.as_str())?;
    out.set_item(
        "worst_normalized_l2_residual",
        diagnostics.worst_normalized_l2_residual,
    )?;
    out.set_item("residual_spread", diagnostics.residual_spread)?;
    let rows = PyList::empty(py);
    for row in &diagnostics.per_model {
        rows.append(operator_model_to_dict(py, row)?)?;
    }
    out.set_item("per_model", rows)?;
    Ok(out)
}

fn operator_model_to_dict<'py>(
    py: Python<'py>,
    row: &BreastUstForwardOperatorModelDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("model", row.model.as_str())?;
    out.set_item("normalized_l2_residual", row.normalized_l2_residual)?;
    out.set_item(
        "row_normalized_l2_residual_mean",
        row.row_normalized_l2_residual_mean,
    )?;
    out.set_item(
        "source_scale_magnitude_coefficient_of_variation",
        row.source_scale_magnitude_coefficient_of_variation,
    )?;
    out.set_item(
        "source_scale_phase_span_rad",
        row.source_scale_phase_span_rad,
    )?;
    Ok(out)
}

fn scattering_increment_to_dict<'py>(
    py: Python<'py>,
    diagnostics: &BreastUstScatteringIncrementDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("model_count", diagnostics.model_count)?;
    out.set_item(
        "receiver_channel_policy",
        diagnostics.receiver_channel_policy.as_str(),
    )?;
    out.set_item(
        "direct_field_normalized_l2_residual",
        diagnostics.direct_field_normalized_l2_residual,
    )?;
    out.set_item(
        "direct_field_scaled_residual_l2_norm",
        diagnostics.direct_field_scaled_residual_l2_norm,
    )?;
    out.set_item(
        "observed_increment_l2_norm",
        diagnostics.observed_increment_l2_norm,
    )?;
    out.set_item("best_model", diagnostics.best_model.as_str())?;
    out.set_item(
        "best_normalized_increment_residual",
        diagnostics.best_normalized_increment_residual,
    )?;
    out.set_item(
        "best_model_scaled_increment_model",
        diagnostics.best_model_scaled_increment_model.as_str(),
    )?;
    out.set_item(
        "best_model_scaled_normalized_increment_residual",
        diagnostics.best_model_scaled_normalized_increment_residual,
    )?;
    out.set_item("worst_model", diagnostics.worst_model.as_str())?;
    out.set_item(
        "worst_normalized_increment_residual",
        diagnostics.worst_normalized_increment_residual,
    )?;
    out.set_item(
        "increment_residual_spread",
        diagnostics.increment_residual_spread,
    )?;
    let rows = PyList::empty(py);
    for row in &diagnostics.per_model {
        rows.append(scattering_increment_model_to_dict(py, row)?)?;
    }
    out.set_item("per_model", rows)?;
    Ok(out)
}

fn scattering_increment_model_to_dict<'py>(
    py: Python<'py>,
    row: &BreastUstScatteringIncrementModelDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("model", row.model.as_str())?;
    out.set_item(
        "predicted_increment_l2_norm",
        row.predicted_increment_l2_norm,
    )?;
    out.set_item("increment_residual_l2_norm", row.increment_residual_l2_norm)?;
    out.set_item(
        "normalized_increment_residual",
        row.normalized_increment_residual,
    )?;
    out.set_item(
        "row_normalized_increment_residual_mean",
        row.row_normalized_increment_residual_mean,
    )?;
    out.set_item(
        "row_normalized_increment_residual_max",
        row.row_normalized_increment_residual_max,
    )?;
    out.set_item("increment_energy_ratio", row.increment_energy_ratio)?;
    out.set_item(
        "baseline_scaled_full_field_normalized_residual",
        row.baseline_scaled_full_field_normalized_residual,
    )?;
    out.set_item(
        "model_scaled_full_field_normalized_residual",
        row.model_scaled_full_field_normalized_residual,
    )?;
    out.set_item(
        "model_scaled_observed_increment_l2_norm",
        row.model_scaled_observed_increment_l2_norm,
    )?;
    out.set_item(
        "model_scaled_increment_residual_l2_norm",
        row.model_scaled_increment_residual_l2_norm,
    )?;
    out.set_item(
        "model_scaled_normalized_increment_residual",
        row.model_scaled_normalized_increment_residual,
    )?;
    out.set_item(
        "model_scaled_increment_energy_ratio",
        row.model_scaled_increment_energy_ratio,
    )?;
    out.set_item(
        "source_scale_relative_drift_mean",
        row.source_scale_relative_drift_mean,
    )?;
    out.set_item(
        "source_scale_relative_drift_max",
        row.source_scale_relative_drift_max,
    )?;
    out.set_item(
        "source_scale_phase_drift_mean_abs_rad",
        row.source_scale_phase_drift_mean_abs_rad,
    )?;
    out.set_item(
        "source_scale_phase_drift_max_abs_rad",
        row.source_scale_phase_drift_max_abs_rad,
    )?;
    Ok(out)
}

fn kwavers_to_value_py(err: kwavers_core::error::KwaversError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

