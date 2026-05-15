//! `run_theranostic_inverse_from_ritk` pyfunction and result serialization.

use kwavers::clinical::therapy::theranostic_guidance::{
    AnatomyKind, TheranosticInverseConfig, WaveformMisfit,
    build_abdominal_placement_context, build_brain_placement_context, placement_metrics,
    prepare_abdominal_slice, prepare_brain_slice, run_theranostic_inverse,
    PlacementContext, THERANOSTIC_OPERATOR_MODEL,
};
use kwavers::solver::inverse::seismic::brain_helmet::{resample_head_slice, select_head_slice};
use ndarray::Array1;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use crate::ritk_image::load_ritk_nifti;
use super::helpers::{
    kwavers_to_py, labels_from_volume, metric_dict, placement_context_skin_gap,
    placement_dict, placement_points3_to_array, point_axis,
};

#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    segmentation_nifti_path = None,
    anatomy = "brain",
    grid_size = 64,
    element_count = None,
    iterations = 12,
    frequencies_hz = None,
    receiver_offsets = None,
    source_pressure_pa = None,
    noise_fraction = 0.012,
    inverse_encoding_rows_per_code = 2,
    waveform_misfit = "charbonnier",
    waveform_misfit_scale_fraction = 0.012
))]
pub fn run_theranostic_inverse_from_ritk<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    segmentation_nifti_path: Option<&str>,
    anatomy: &str,
    grid_size: usize,
    element_count: Option<usize>,
    iterations: usize,
    frequencies_hz: Option<Vec<f64>>,
    receiver_offsets: Option<Vec<usize>>,
    source_pressure_pa: Option<f64>,
    noise_fraction: f64,
    inverse_encoding_rows_per_code: usize,
    waveform_misfit: &str,
    waveform_misfit_scale_fraction: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let anatomy = AnatomyKind::from_name(anatomy).map_err(kwavers_to_py)?;
    let (mut ct, spacing_mm) = load_ritk_nifti(Path::new(ct_nifti_path))?;
    ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
    let mut config = TheranosticInverseConfig::new(anatomy);
    config.grid_size = grid_size;
    config.iterations = iterations;
    config.noise_fraction = noise_fraction;
    config.inverse_encoding_rows_per_code = inverse_encoding_rows_per_code;
    config.waveform_misfit = WaveformMisfit::from_name(waveform_misfit)
        .ok_or_else(|| PyValueError::new_err("waveform_misfit must be 'charbonnier' or 'l2'"))?;
    config.waveform_misfit_scale_fraction = waveform_misfit_scale_fraction;
    if let Some(count) = element_count {
        config.element_count = count;
    }
    if let Some(freqs) = frequencies_hz {
        config.frequencies_hz = freqs;
    }
    if let Some(offsets) = receiver_offsets {
        config.receiver_offsets = offsets;
    }
    if let Some(pressure) = source_pressure_pa {
        config.source_pressure_pa = pressure;
    }

    let (prepared, placement_context) = match anatomy {
        AnatomyKind::Brain => {
            let selected = select_head_slice(&ct).map_err(kwavers_to_py)?;
            let resampled =
                resample_head_slice(&ct, spacing_mm, selected, grid_size).map_err(kwavers_to_py)?;
            let placement_context =
                build_brain_placement_context(&ct, spacing_mm, selected, &config)
                    .map_err(kwavers_to_py)?;
            (
                prepare_brain_slice(resampled.hu, resampled.spacing_m, selected)
                    .map_err(kwavers_to_py)?,
                placement_context,
            )
        }
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            let seg_path = segmentation_nifti_path.ok_or_else(|| {
                PyValueError::new_err("segmentation_nifti_path is required for liver and kidney")
            })?;
            let (seg, _) = load_ritk_nifti(Path::new(seg_path))?;
            let labels = labels_from_volume(seg);
            let placement_context =
                build_abdominal_placement_context(anatomy, &ct, &labels, spacing_mm, &config)
                    .map_err(kwavers_to_py)?;
            (
                prepare_abdominal_slice(anatomy, &ct, &labels, spacing_mm, grid_size)
                    .map_err(kwavers_to_py)?,
                placement_context,
            )
        }
    };
    let result = py
        .detach(|| run_theranostic_inverse(prepared, &config))
        .map_err(kwavers_to_py)?;
    result_to_dict(py, result, &config, placement_context)
}

fn result_to_dict<'py>(
    py: Python<'py>,
    result: kwavers::clinical::therapy::theranostic_guidance::TheranosticInverseResult,
    config: &TheranosticInverseConfig,
    placement_context: PlacementContext,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let operator_backend = result.operator_backend.clone();
    let operator_storage_values = result.operator_storage_values;
    let dense_operator_values = result.dense_operator_values;
    let inverse_model_family = result.inverse_model_family.clone();
    let is_full_wave_inversion = result.is_full_wave_inversion;
    let uses_nonlinear_wave_propagation = result.uses_nonlinear_wave_propagation;
    let prepared = result.prepared;
    let layout = result.layout;
    let placement = placement_metrics(&layout, &prepared.body_mask, prepared.spacing_m);
    let placement_context_model = placement_context.model_name.clone();
    let placement_spacing_m = (placement_context.spacing_x_m, placement_context.spacing_y_m);
    let placement_focus_m = (
        placement_context.focus_m.x_m,
        placement_context.focus_m.y_m,
        placement_context.focus_m.z_m,
    );
    let placement_skin_contact_m = (
        placement_context.skin_contact_m.x_m,
        placement_context.skin_contact_m.y_m,
        placement_context.skin_contact_m.z_m,
    );
    let placement_context_skin_gap_m = placement_context_skin_gap(&placement_context);
    let placement_context_surface_points = placement_context.body_surface_points_m.len();
    let placement_therapy_points = placement_points3_to_array(&placement_context.therapy_points_m);
    let placement_imaging_points = placement_points3_to_array(&placement_context.imaging_points_m);
    let placement_body_surface_points =
        placement_points3_to_array(&placement_context.body_surface_points_m);
    out.set_item("anatomy", prepared.anatomy.label())?;
    out.set_item("device_model", layout.model_name.clone())?;
    out.set_item("ct_hu", prepared.ct_hu.into_pyarray(py))?;
    out.set_item("label", prepared.label.into_pyarray(py))?;
    out.set_item("sound_speed_m_s", prepared.sound_speed_m_s.into_pyarray(py))?;
    out.set_item(
        "attenuation_np_per_m_mhz",
        prepared.attenuation_np_per_m_mhz.into_pyarray(py),
    )?;
    out.set_item("body_mask", prepared.body_mask.into_pyarray(py))?;
    out.set_item("organ_mask", prepared.organ_mask.into_pyarray(py))?;
    out.set_item("target_mask", prepared.target_mask.into_pyarray(py))?;
    out.set_item("exposure", result.exposure.into_pyarray(py))?;
    out.set_item("lesion_target", result.lesion_target.into_pyarray(py))?;
    out.set_item(
        "anatomy_reconstruction",
        result.anatomy_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "active_lesion_reconstruction",
        result.active_lesion_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "waveform_rtm_reconstruction",
        result.waveform_rtm_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "subharmonic_reconstruction",
        result.subharmonic_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "harmonic_reconstruction",
        result.harmonic_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "ultraharmonic_reconstruction",
        result.ultraharmonic_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "fused_reconstruction",
        result.fused_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "therapy_x_m",
        point_axis(&layout.therapy_elements, true).into_pyarray(py),
    )?;
    out.set_item(
        "therapy_y_m",
        point_axis(&layout.therapy_elements, false).into_pyarray(py),
    )?;
    out.set_item(
        "imaging_x_m",
        point_axis(&layout.imaging_receivers, true).into_pyarray(py),
    )?;
    out.set_item(
        "imaging_y_m",
        point_axis(&layout.imaging_receivers, false).into_pyarray(py),
    )?;
    out.set_item("focus_m", (layout.focus_m.x_m, layout.focus_m.y_m))?;
    out.set_item(
        "skin_contact_m",
        (layout.skin_contact_m.x_m, layout.skin_contact_m.y_m),
    )?;
    out.set_item("spacing_m", prepared.spacing_m)?;
    out.set_item("source_slice_index", prepared.source_slice_index)?;
    out.set_item("element_count", config.element_count)?;
    out.set_item("frequencies_hz", config.frequencies_hz.clone())?;
    out.set_item("receiver_offsets", config.receiver_offsets.clone())?;
    out.set_item("source_pressure_pa", config.source_pressure_pa)?;
    out.set_item("geometry_model", layout.model_name.clone())?;
    out.set_item("placement_metrics", placement_dict(py, &placement)?)?;
    out.set_item("placement_context_model", placement_context_model)?;
    out.set_item("placement_ct_hu", placement_context.ct_hu.into_pyarray(py))?;
    out.set_item(
        "placement_body_mask",
        placement_context.body_mask.into_pyarray(py),
    )?;
    out.set_item(
        "placement_target_mask",
        placement_context.target_mask.into_pyarray(py),
    )?;
    out.set_item("placement_spacing_m", placement_spacing_m)?;
    out.set_item("placement_slice_index", placement_context.slice_index)?;
    out.set_item(
        "placement_therapy_points_m",
        placement_therapy_points.into_pyarray(py),
    )?;
    out.set_item(
        "placement_imaging_points_m",
        placement_imaging_points.into_pyarray(py),
    )?;
    out.set_item(
        "placement_body_surface_points_m",
        placement_body_surface_points.into_pyarray(py),
    )?;
    out.set_item("placement_focus_m", placement_focus_m)?;
    out.set_item("placement_skin_contact_m", placement_skin_contact_m)?;
    out.set_item("placement_context_skin_gap_m", placement_context_skin_gap_m)?;
    out.set_item(
        "placement_context_surface_points",
        placement_context_surface_points,
    )?;
    out.set_item("operator_model", THERANOSTIC_OPERATOR_MODEL)?;
    out.set_item("operator_backend", operator_backend)?;
    out.set_item("operator_storage_values", operator_storage_values)?;
    out.set_item("dense_operator_values", dense_operator_values)?;
    out.set_item("inverse_model_family", inverse_model_family)?;
    out.set_item("is_full_wave_inversion", is_full_wave_inversion)?;
    out.set_item(
        "uses_nonlinear_wave_propagation",
        uses_nonlinear_wave_propagation,
    )?;
    out.set_item("waveform_model", result.waveform.model_name)?;
    out.set_item("waveform_misfit", result.waveform.misfit_name)?;
    out.set_item("waveform_misfit_scale", result.waveform.misfit_scale)?;
    out.set_item("waveform_objective", result.waveform.objective_value)?;
    out.set_item("waveform_residual_energy", result.waveform.residual_energy)?;
    out.set_item("waveform_observed_energy", result.waveform.observed_energy)?;
    out.set_item("waveform_receiver_count", result.waveform.receiver_count)?;
    out.set_item("waveform_time_steps", result.waveform.time_steps)?;
    out.set_item("waveform_dt_s", result.waveform.dt_s)?;
    out.set_item("measurements", result.measurements)?;
    out.set_item("encoded_measurements", result.encoded_measurements)?;
    out.set_item("unencoded_measurements", result.unencoded_measurements)?;
    out.set_item(
        "inverse_encoding_rows_per_code",
        result.inverse_encoding_rows_per_code,
    )?;
    out.set_item("active_voxels", result.active_voxels)?;
    out.set_item(
        "objective_history",
        Array1::from(result.objective_history).into_pyarray(py),
    )?;
    let metrics = PyDict::new(py);
    metrics.set_item("anatomy", metric_dict(py, &result.anatomy_metrics)?)?;
    metrics.set_item("active_lesion", metric_dict(py, &result.active_metrics)?)?;
    metrics.set_item("waveform_rtm", metric_dict(py, &result.waveform_metrics)?)?;
    metrics.set_item("subharmonic", metric_dict(py, &result.subharmonic_metrics)?)?;
    metrics.set_item("harmonic", metric_dict(py, &result.harmonic_metrics)?)?;
    metrics.set_item(
        "ultraharmonic",
        metric_dict(py, &result.ultraharmonic_metrics)?,
    )?;
    metrics.set_item("fusion", metric_dict(py, &result.fused_metrics)?)?;
    out.set_item("metrics", metrics)?;
    Ok(out)
}
