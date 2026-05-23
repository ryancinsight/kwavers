//! PyO3 wrapper for homogeneous direct-field breast-FWI diagnostics.

use super::{kwavers_to_py, PyBreastFwiPstdDatasetConfig, PyMultiRowRingArray};
use kwavers::clinical::imaging::reconstruction::breast_ust_fwi::{
    diagnose_breast_ust_homogeneous_direct_field, BreastUstDirectFieldDiagnostics,
    BreastUstHomogeneousDirectFieldDiagnostics,
};
use numpy::PyReadonlyArray3;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

#[pyfunction]
pub fn diagnose_breast_fwi_homogeneous_direct_field<'py>(
    py: Python<'py>,
    homogeneous_sound_speed_m_s: PyReadonlyArray3<'py, f64>,
    array: &PyMultiRowRingArray,
    frequencies_hz: Vec<f64>,
    config: &PyBreastFwiPstdDatasetConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let sound_speed = homogeneous_sound_speed_m_s.as_array().to_owned();
    let diagnostics = py
        .detach(|| {
            diagnose_breast_ust_homogeneous_direct_field(
                &sound_speed,
                &array.inner,
                &frequencies_hz,
                config.inner,
            )
        })
        .map_err(kwavers_to_py)?;
    direct_field_report_to_dict(py, &diagnostics)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        diagnose_breast_fwi_homogeneous_direct_field,
        m
    )?)?;
    Ok(())
}

fn direct_field_report_to_dict<'py>(
    py: Python<'py>,
    diagnostics: &BreastUstHomogeneousDirectFieldDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = direct_field_metrics_to_dict(py, &diagnostics.point_source)?;
    out.set_item(
        "source_kappa_filtered",
        direct_field_metrics_to_dict(py, &diagnostics.source_kappa_filtered)?,
    )?;
    out.set_item(
        "source_kappa_filtered_residual_delta",
        diagnostics.source_kappa_filtered_residual_delta,
    )?;
    out.set_item(
        "source_kappa_filtered_passive_residual_delta",
        diagnostics.source_kappa_filtered_passive_residual_delta,
    )?;
    out.set_item(
        "pstd_periodic",
        direct_field_metrics_to_dict(py, &diagnostics.pstd_periodic)?,
    )?;
    out.set_item(
        "pstd_periodic_residual_delta",
        diagnostics.pstd_periodic_residual_delta,
    )?;
    out.set_item(
        "pstd_periodic_passive_residual_delta",
        diagnostics.pstd_periodic_passive_residual_delta,
    )?;
    out.set_item("model_family", diagnostics.model_family)?;
    Ok(out)
}

fn direct_field_metrics_to_dict<'py>(
    py: Python<'py>,
    metrics: &BreastUstDirectFieldDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("normalized_l2_residual", metrics.normalized_l2_residual)?;
    out.set_item(
        "row_normalized_l2_residual_mean",
        metrics.row_normalized_l2_residual_mean,
    )?;
    out.set_item(
        "active_only_normalized_l2_residual",
        metrics.active_only_normalized_l2_residual,
    )?;
    out.set_item(
        "passive_only_normalized_l2_residual",
        metrics.passive_only_normalized_l2_residual,
    )?;
    out.set_item(
        "source_scale_magnitude_coefficient_of_variation",
        metrics.source_scale_magnitude_coefficient_of_variation,
    )?;
    out.set_item(
        "source_scale_phase_span_rad",
        metrics.source_scale_phase_span_rad,
    )?;
    out.set_item("active_pair_count", metrics.active_pair_count)?;
    out.set_item(
        "active_self_channel_phase_error_rms_rad",
        metrics.active_self_channel_phase_error_rms_rad,
    )?;
    out.set_item(
        "active_self_channel_phase_error_max_abs_rad",
        metrics.active_self_channel_phase_error_max_abs_rad,
    )?;
    out.set_item(
        "active_self_channel_log_amplitude_error_rms",
        metrics.active_self_channel_log_amplitude_error_rms,
    )?;
    out.set_item("passive_pair_count", metrics.passive_pair_count)?;
    out.set_item("passive_range_min_m", metrics.passive_range_min_m)?;
    out.set_item("passive_range_max_m", metrics.passive_range_max_m)?;
    out.set_item(
        "passive_phase_error_rms_rad",
        metrics.passive_phase_error_rms_rad,
    )?;
    out.set_item(
        "passive_phase_error_max_abs_rad",
        metrics.passive_phase_error_max_abs_rad,
    )?;
    out.set_item(
        "passive_log_amplitude_error_rms",
        metrics.passive_log_amplitude_error_rms,
    )?;
    Ok(out)
}
