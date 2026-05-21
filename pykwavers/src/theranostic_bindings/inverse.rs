//! `run_theranostic_inverse_from_ritk` pyfunction and result serialization.

use kwavers::clinical::imaging::reconstruction::transcranial_ust::{
    resample_head_slice, select_head_slice,
};
use kwavers::clinical::therapy::theranostic_guidance::{
    build_abdominal_placement_context, build_brain_placement_context, placement_metrics,
    prepare_abdominal_slice, prepare_brain_slice, run_theranostic_inverse,
    synthetic::{
        synthetic_abdominal_kidney_phantom, synthetic_abdominal_liver_phantom,
        synthetic_brain_phantom,
    },
    target_index_from_mask_fraction_3d, AnatomyKind, BrainTargetSelection, PlacementContext,
    TheranosticInverseConfig, TransmitScheduleConfig, TransmitScheduleStrategy, WaveformMisfit,
    THERANOSTIC_OPERATOR_MODEL, TRANSMIT_SCHEDULE_MODEL,
};
use ndarray::{Array1, Array3};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use super::helpers::{
    kwavers_to_py, labels_from_volume, metric_dict, placement_context_skin_gap, placement_dict,
    point_axis, points3_to_array,
};
use crate::ritk_image::load_ritk_nifti;

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
    target_fraction_xyz = None,
    noise_fraction = 0.012,
    inverse_encoding_rows_per_code = 2,
    transmit_schedule_strategy = "full",
    transmit_budget = None,
    elastic_fwi_iterations = 3,
    waveform_misfit = "charbonnier",
    waveform_misfit_scale_fraction = 0.012
))]
#[allow(clippy::too_many_arguments)]
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
    target_fraction_xyz: Option<(f64, f64, f64)>,
    noise_fraction: f64,
    inverse_encoding_rows_per_code: usize,
    transmit_schedule_strategy: &str,
    transmit_budget: Option<usize>,
    elastic_fwi_iterations: usize,
    waveform_misfit: &str,
    waveform_misfit_scale_fraction: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let anatomy = AnatomyKind::from_name(anatomy).map_err(kwavers_to_py)?;
    let ct_path = Path::new(ct_nifti_path);
    let (ct, spacing_mm) = if ct_path.exists() {
        let (mut ct, spacing_mm) = load_ritk_nifti(ct_path)?;
        ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
        (ct, spacing_mm)
    } else {
        match anatomy {
            AnatomyKind::Brain => synthetic_brain_phantom(),
            AnatomyKind::Liver => {
                let (ct, _, spacing_mm) = synthetic_abdominal_liver_phantom();
                (ct, spacing_mm)
            }
            AnatomyKind::Kidney => {
                let (ct, _, spacing_mm) = synthetic_abdominal_kidney_phantom();
                (ct, spacing_mm)
            }
        }
    };
    let mut config = TheranosticInverseConfig::new(anatomy);
    config.grid_size = grid_size;
    config.iterations = iterations;
    config.noise_fraction = noise_fraction;
    config.inverse_encoding_rows_per_code = inverse_encoding_rows_per_code;
    config.elastic_fwi_iterations = elastic_fwi_iterations;
    config.transmit_schedule = TransmitScheduleConfig {
        strategy: TransmitScheduleStrategy::from_name(transmit_schedule_strategy)
            .map_err(kwavers_to_py)?,
        budget: transmit_budget,
    };
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
            let target_fraction = target_fraction_xyz.map(|(x, y, z)| [x, y, z]);
            let target_index = if let Some(fraction) = target_fraction {
                Some(brain_target_index(&ct, fraction).map_err(kwavers_to_py)?)
            } else {
                None
            };
            let selected = if let Some(index) = target_index {
                index[2]
            } else {
                select_head_slice(&ct).map_err(kwavers_to_py)?
            };
            let resampled =
                resample_head_slice(&ct, spacing_mm, selected, grid_size).map_err(kwavers_to_py)?;
            let crop_bounds_index = resampled.crop_bounds_index;
            let source_dimensions = resampled.source_dimensions;
            let source_spacing_m = resampled.source_spacing_m;
            let target_selection =
                target_index.map_or(BrainTargetSelection::OrganCentroid, |index| {
                    BrainTargetSelection::ResampledIndex(resampled_crop_index_xy(
                        index,
                        crop_bounds_index,
                        grid_size,
                    ))
                });
            let placement_context =
                build_brain_placement_context(&ct, spacing_mm, selected, &config, target_fraction)
                    .map_err(kwavers_to_py)?;
            let mut prepared = prepare_brain_slice(
                resampled.hu,
                resampled.spacing_m,
                selected,
                target_selection,
            )
            .map_err(kwavers_to_py)?;
            prepared.source_dimensions = source_dimensions;
            prepared.source_spacing_m = source_spacing_m;
            prepared.crop_bounds_index = crop_bounds_index;
            (prepared, placement_context)
        }
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            // Load segmentation labels: from NIfTI when available, else synthetic.
            let labels = if let Some(seg_path_str) = segmentation_nifti_path {
                let seg_path = Path::new(seg_path_str);
                if seg_path.exists() {
                    let (seg, _) = load_ritk_nifti(seg_path)?;
                    labels_from_volume(seg)
                } else {
                    // Paired synthetic segmentation matching the CT phantom.
                    match anatomy {
                        AnatomyKind::Liver => synthetic_abdominal_liver_phantom().1,
                        AnatomyKind::Kidney => synthetic_abdominal_kidney_phantom().1,
                        _ => unreachable!(),
                    }
                }
            } else if !ct_path.exists() {
                // No seg path provided and CT is also synthetic: use synthetic label.
                match anatomy {
                    AnatomyKind::Liver => synthetic_abdominal_liver_phantom().1,
                    AnatomyKind::Kidney => synthetic_abdominal_kidney_phantom().1,
                    _ => unreachable!(),
                }
            } else {
                return Err(PyValueError::new_err(
                    "segmentation_nifti_path is required for liver and kidney when using real CT",
                ));
            };
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
    result_to_dict(py, result, &config, placement_context, target_fraction_xyz)
}

fn brain_target_index(
    ct_volume_hu: &Array3<f64>,
    fraction: [f64; 3],
) -> kwavers::core::error::KwaversResult<[usize; 3]> {
    let brain = ct_volume_hu.mapv(|hu| hu > -300.0 && hu < 300.0);
    if brain.iter().any(|active| *active) {
        target_index_from_mask_fraction_3d(&brain, fraction)
    } else {
        let body = ct_volume_hu.mapv(|hu| hu > -300.0);
        target_index_from_mask_fraction_3d(&body, fraction)
    }
}

fn resampled_crop_index_xy(
    source_index: [usize; 3],
    crop_bounds_index: [usize; 4],
    grid_size: usize,
) -> [f64; 2] {
    let scale = (grid_size - 1) as f64;
    let x0 = crop_bounds_index[0] as f64;
    let x1 = crop_bounds_index[1] as f64;
    let y0 = crop_bounds_index[2] as f64;
    let y1 = crop_bounds_index[3] as f64;
    [
        (source_index[0] as f64 - x0) * scale / (x1 - x0).max(1.0),
        (source_index[1] as f64 - y0) * scale / (y1 - y0).max(1.0),
    ]
}

fn result_to_dict<'py>(
    py: Python<'py>,
    result: kwavers::clinical::therapy::theranostic_guidance::TheranosticInverseResult,
    config: &TheranosticInverseConfig,
    placement_context: PlacementContext,
    target_fraction_xyz: Option<(f64, f64, f64)>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let operator_backend = result.operator_backend.clone();
    let operator_storage_values = result.operator_storage_values;
    let dense_operator_values = result.dense_operator_values;
    let inverse_model_family = result.inverse_model_family.clone();
    let exposure_model = result.exposure_model.clone();
    let exposure_backend = result.exposure_backend.clone();
    let exposure_uses_hybrid_pstd_fdtd = result.exposure_uses_hybrid_pstd_fdtd;
    let exposure_source_count = result.exposure_source_count;
    let exposure_time_steps = result.exposure_time_steps;
    let exposure_dt_s = result.exposure_dt_s;
    let exposure_workspace_values = result.exposure_workspace_values;
    let elastic_shear_model = result.elastic_shear_model.clone();
    let elastic_shear_receiver_count = result.elastic_shear.receiver_count;
    let elastic_shear_time_steps = result.elastic_shear.time_steps;
    let elastic_shear_dt_s = result.elastic_shear.dt_s;
    let elastic_shear_iteration_count = result.elastic_shear.iteration_count;
    let elastic_shear_accepted_step_count = result.elastic_shear.accepted_step_count;
    let elastic_shear_objective_history = result.elastic_shear.objective_history.clone();
    let elastic_shear_baseline_trace_energy = result.elastic_shear.baseline_trace_energy;
    let elastic_shear_lesion_trace_energy = result.elastic_shear.lesion_trace_energy;
    let elastic_shear_residual_trace_energy = result.elastic_shear.residual_trace_energy;
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
    let placement_therapy_points = points3_to_array(&placement_context.therapy_points_m);
    let placement_imaging_points = points3_to_array(&placement_context.imaging_points_m);
    let placement_body_surface_points = points3_to_array(&placement_context.body_surface_points_m);
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
    out.set_item(
        "exposure_raw_peak_pressure",
        result.exposure_raw_peak_pressure.into_pyarray(py),
    )?;
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
        "elastic_shear_reconstruction",
        result.elastic_shear_reconstruction.into_pyarray(py),
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
    out.set_item("source_dimensions", prepared.source_dimensions)?;
    out.set_item("source_spacing_m", prepared.source_spacing_m)?;
    out.set_item("crop_bounds_index", prepared.crop_bounds_index)?;
    out.set_item("element_count", config.element_count)?;
    out.set_item("frequencies_hz", config.frequencies_hz.clone())?;
    out.set_item("receiver_offsets", config.receiver_offsets.clone())?;
    out.set_item("source_pressure_pa", config.source_pressure_pa)?;
    out.set_item("transmit_schedule_model", TRANSMIT_SCHEDULE_MODEL)?;
    out.set_item(
        "transmit_schedule_strategy",
        result.transmit_schedule.strategy.label(),
    )?;
    out.set_item(
        "transmit_budget_requested",
        result.transmit_schedule.requested_budget,
    )?;
    out.set_item(
        "transmit_budget_effective",
        result.transmit_schedule.effective_budget(),
    )?;
    out.set_item(
        "transmit_budget_fraction",
        result.transmit_schedule.budget_fraction(),
    )?;
    out.set_item(
        "transmit_sequence_indices",
        result.transmit_schedule.active_indices.clone(),
    )?;
    out.set_item(
        "elastic_frequencies_hz",
        config.elastic_frequencies_hz.clone(),
    )?;
    out.set_item("elastic_shear_speed_m_s", config.elastic_shear_speed_m_s)?;
    out.set_item("elastic_fwi_iterations", config.elastic_fwi_iterations)?;
    if let Some((x, y, z)) = target_fraction_xyz {
        out.set_item("target_fraction_xyz", (x, y, z))?;
    }
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
    out.set_item("exposure_model", exposure_model)?;
    out.set_item("exposure_backend", exposure_backend)?;
    out.set_item(
        "exposure_uses_hybrid_pstd_fdtd",
        exposure_uses_hybrid_pstd_fdtd,
    )?;
    out.set_item("exposure_source_count", exposure_source_count)?;
    out.set_item("exposure_time_steps", exposure_time_steps)?;
    out.set_item("exposure_dt_s", exposure_dt_s)?;
    out.set_item("exposure_workspace_values", exposure_workspace_values)?;
    out.set_item("elastic_shear_model", elastic_shear_model)?;
    out.set_item("elastic_shear_receiver_count", elastic_shear_receiver_count)?;
    out.set_item("elastic_shear_time_steps", elastic_shear_time_steps)?;
    out.set_item("elastic_shear_dt_s", elastic_shear_dt_s)?;
    out.set_item(
        "elastic_shear_iteration_count",
        elastic_shear_iteration_count,
    )?;
    out.set_item(
        "elastic_shear_accepted_step_count",
        elastic_shear_accepted_step_count,
    )?;
    out.set_item(
        "elastic_shear_objective_history",
        elastic_shear_objective_history,
    )?;
    out.set_item(
        "elastic_shear_baseline_trace_energy",
        elastic_shear_baseline_trace_energy,
    )?;
    out.set_item(
        "elastic_shear_lesion_trace_energy",
        elastic_shear_lesion_trace_energy,
    )?;
    out.set_item(
        "elastic_shear_residual_trace_energy",
        elastic_shear_residual_trace_energy,
    )?;
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
    metrics.set_item(
        "elastic_shear",
        metric_dict(py, &result.elastic_shear_metrics)?,
    )?;
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
