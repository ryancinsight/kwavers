//! Python bindings for same-device therapy/imaging inverse simulations.

use kwavers::clinical::therapy::theranostic_guidance::{
    build_abdominal_placement_context, build_brain_placement_context, placement_metrics,
    plan_brain_helmet_placement, prepare_abdominal_slice, prepare_brain_slice,
    run_theranostic_inverse, run_theranostic_nonlinear_3d, AnatomyKind, DevicePlacementMetrics,
    Nonlinear3dConfig, PlacementContext, PlacementPoint3, Point3, ReconstructionMetrics,
    TheranosticInverseConfig, VolumeReconstructionMetrics, WaveformMisfit,
    THERANOSTIC_OPERATOR_MODEL,
};
use kwavers::solver::inverse::seismic::brain_helmet::{resample_head_slice, select_head_slice};
use ndarray::{Array1, Array2, Array3};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;
use std::path::Path;

use crate::ritk_image::load_ritk_nifti;

#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    segmentation_nifti_path = None,
    anatomy = "brain",
    grid_size = 24,
    element_count = None,
    receiver_count = 48,
    source_encoding_count = 3,
    checkpoint_interval_steps = 128,
    iterations = 2,
    frequency_hz = None,
    source_pressure_pa = None,
    cycles = 3.0,
    lesion_delta_c_m_s = -35.0,
    lesion_delta_beta = 0.85,
    sound_speed_regularization = 2.0e-3,
    nonlinearity_regularization = 1.0e-3,
    gradient_smoothing_steps = 2,
    bubble_radius_m = 2.0e-6,
    bubble_time_steps_per_period = 96,
    cavitation_iterations = 24,
    cavitation_regularization = 1.0e-4
))]
fn run_theranostic_nonlinear_3d_from_ritk<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    segmentation_nifti_path: Option<&str>,
    anatomy: &str,
    grid_size: usize,
    element_count: Option<usize>,
    receiver_count: usize,
    source_encoding_count: usize,
    checkpoint_interval_steps: usize,
    iterations: usize,
    frequency_hz: Option<f64>,
    source_pressure_pa: Option<f64>,
    cycles: f64,
    lesion_delta_c_m_s: f64,
    lesion_delta_beta: f64,
    sound_speed_regularization: f64,
    nonlinearity_regularization: f64,
    gradient_smoothing_steps: usize,
    bubble_radius_m: f64,
    bubble_time_steps_per_period: usize,
    cavitation_iterations: usize,
    cavitation_regularization: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let anatomy = AnatomyKind::from_name(anatomy).map_err(kwavers_to_py)?;
    let (mut ct, spacing_mm) = load_ritk_nifti(Path::new(ct_nifti_path))?;
    ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
    let labels = if let Some(path) = segmentation_nifti_path {
        let (seg, _) = load_ritk_nifti(Path::new(path))?;
        Some(labels_from_volume(seg))
    } else if matches!(anatomy, AnatomyKind::Liver | AnatomyKind::Kidney) {
        return Err(PyValueError::new_err(
            "segmentation_nifti_path is required for nonlinear liver and kidney simulations",
        ));
    } else {
        None
    };
    let mut config = Nonlinear3dConfig::new(anatomy);
    config.grid_size = grid_size;
    config.receiver_count = receiver_count;
    config.source_encoding_count = source_encoding_count;
    config.checkpoint_interval_steps = checkpoint_interval_steps;
    config.iterations = iterations;
    config.cycles = cycles;
    config.lesion_delta_c_m_s = lesion_delta_c_m_s;
    config.lesion_delta_beta = lesion_delta_beta;
    config.sound_speed_regularization = sound_speed_regularization;
    config.nonlinearity_regularization = nonlinearity_regularization;
    config.gradient_smoothing_steps = gradient_smoothing_steps;
    config.bubble_radius_m = bubble_radius_m;
    config.bubble_time_steps_per_period = bubble_time_steps_per_period;
    config.cavitation_iterations = cavitation_iterations;
    config.cavitation_regularization = cavitation_regularization;
    if let Some(count) = element_count {
        config.element_count = count;
    }
    if let Some(frequency) = frequency_hz {
        config.frequency_hz = frequency;
    }
    if let Some(pressure) = source_pressure_pa {
        config.source_pressure_pa = pressure;
    }
    let result = py
        .detach(|| run_theranostic_nonlinear_3d(anatomy, &ct, labels.as_ref(), spacing_mm, &config))
        .map_err(kwavers_to_py)?;
    nonlinear3d_result_to_dict(py, result, &config)
}

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
fn run_theranostic_inverse_from_ritk<'py>(
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

#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    element_count = 1024,
    surface_stride = 6,
    body_hu_threshold = -350.0,
    skull_hu_threshold = 300.0
))]
fn plan_brain_helmet_placement_from_ritk_ct<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    element_count: usize,
    surface_stride: usize,
    body_hu_threshold: f64,
    skull_hu_threshold: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (mut ct, spacing_mm) = load_ritk_nifti(Path::new(ct_nifti_path))?;
    ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
    let placement = py
        .detach(|| {
            plan_brain_helmet_placement(
                &ct,
                spacing_mm,
                element_count,
                surface_stride,
                body_hu_threshold,
                skull_hu_threshold,
            )
        })
        .map_err(kwavers_to_py)?;
    let out = PyDict::new(py);
    out.set_item(
        "head_surface_points_m",
        points3_to_array(&placement.head_surface_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "skull_surface_points_m",
        points3_to_array(&placement.skull_surface_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "therapy_elements_m",
        points3_to_array(&placement.therapy_elements_m).into_pyarray(py),
    )?;
    out.set_item(
        "beam_start_points_m",
        points3_to_array(&placement.beam_start_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "beam_end_points_m",
        points3_to_array(&placement.beam_end_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "skull_intersections_m",
        points3_to_array(&placement.skull_intersections_m).into_pyarray(py),
    )?;
    out.set_item(
        "focus_m",
        (
            placement.focus_m.x_m,
            placement.focus_m.y_m,
            placement.focus_m.z_m,
        ),
    )?;
    out.set_item("helmet_radius_m", placement.helmet_radius_m)?;
    out.set_item("intersection_fraction", placement.intersection_fraction)?;
    out.set_item("element_count", element_count)?;
    out.set_item("surface_stride", surface_stride)?;
    out.set_item("body_hu_threshold", body_hu_threshold)?;
    out.set_item("skull_hu_threshold", skull_hu_threshold)?;
    out.set_item(
        "geometry_model",
        "ct_derived_calvarium_1024_element_helmet_with_skull_intersections",
    )?;
    Ok(out)
}

fn labels_from_volume(volume: Array3<f64>) -> Array3<i16> {
    volume.mapv(|value| value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
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

fn nonlinear3d_result_to_dict<'py>(
    py: Python<'py>,
    result: kwavers::clinical::therapy::theranostic_guidance::Nonlinear3dResult,
    config: &Nonlinear3dConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("anatomy", config.anatomy.label())?;
    out.set_item("ct_hu", result.ct_hu.into_pyarray(py))?;
    out.set_item("label", result.label.into_pyarray(py))?;
    out.set_item("body_mask", result.body_mask.into_pyarray(py))?;
    out.set_item("target_mask", result.target_mask.into_pyarray(py))?;
    out.set_item("inversion_mask", result.inversion_mask.into_pyarray(py))?;
    out.set_item(
        "background_sound_speed_m_s",
        result.background_sound_speed_m_s.into_pyarray(py),
    )?;
    out.set_item(
        "true_sound_speed_m_s",
        result.true_sound_speed_m_s.into_pyarray(py),
    )?;
    out.set_item(
        "reconstructed_sound_speed_m_s",
        result.reconstructed_sound_speed_m_s.into_pyarray(py),
    )?;
    out.set_item(
        "reconstructed_delta_c_m_s",
        result.reconstructed_delta_c_m_s.into_pyarray(py),
    )?;
    out.set_item("background_beta", result.background_beta.into_pyarray(py))?;
    out.set_item("true_beta", result.true_beta.into_pyarray(py))?;
    out.set_item(
        "reconstructed_beta",
        result.reconstructed_beta.into_pyarray(py),
    )?;
    out.set_item(
        "reconstructed_delta_beta",
        result.reconstructed_delta_beta.into_pyarray(py),
    )?;
    out.set_item(
        "multiparameter_fwi_score",
        result.multiparameter_fwi_score.into_pyarray(py),
    )?;
    out.set_item(
        "nonlinear_fusion_score",
        result.nonlinear_fusion_score.into_pyarray(py),
    )?;
    out.set_item(
        "westervelt_peak_pressure_pa",
        result.westervelt_peak_pressure_pa.into_pyarray(py),
    )?;
    out.set_item(
        "cavitation_source_density",
        result.cavitation_source_density.into_pyarray(py),
    )?;
    out.set_item(
        "reconstructed_cavitation_density",
        result.reconstructed_cavitation_density.into_pyarray(py),
    )?;
    out.set_item(
        "fwi_objective_history",
        Array1::from(result.fwi_objective_history).into_pyarray(py),
    )?;
    out.set_item(
        "cavitation_objective_history",
        Array1::from(result.cavitation_objective_history).into_pyarray(py),
    )?;
    out.set_item(
        "therapy_points_m",
        points3_to_array(&result.therapy_points_m).into_pyarray(py),
    )?;
    out.set_item(
        "receiver_points_m",
        points3_to_array(&result.receiver_points_m).into_pyarray(py),
    )?;
    out.set_item("spacing_m", result.spacing_m)?;
    out.set_item("dt_s", result.dt_s)?;
    out.set_item("time_steps", result.time_steps)?;
    out.set_item("active_voxels", result.active_voxels)?;
    out.set_item("grid_size", config.grid_size)?;
    out.set_item("element_count", config.element_count)?;
    out.set_item("receiver_count", config.receiver_count)?;
    out.set_item("source_encoding_count", config.source_encoding_count)?;
    out.set_item(
        "checkpoint_interval_steps",
        config.checkpoint_interval_steps,
    )?;
    out.set_item("frequency_hz", config.frequency_hz)?;
    out.set_item("source_pressure_pa", config.source_pressure_pa)?;
    out.set_item("cycles", config.cycles)?;
    out.set_item("lesion_delta_c_m_s", config.lesion_delta_c_m_s)?;
    out.set_item("lesion_delta_beta", config.lesion_delta_beta)?;
    out.set_item(
        "sound_speed_regularization",
        config.sound_speed_regularization,
    )?;
    out.set_item(
        "nonlinearity_regularization",
        config.nonlinearity_regularization,
    )?;
    out.set_item("gradient_smoothing_steps", config.gradient_smoothing_steps)?;
    out.set_item("bubble_radius_m", config.bubble_radius_m)?;
    out.set_item("aperture_model", result.aperture_model)?;
    out.set_item("model_family", result.model_family)?;
    out.set_item("propagator_model", result.propagator_model)?;
    out.set_item("cavitation_inverse_model", result.cavitation_inverse_model)?;
    out.set_item("is_full_wave_inversion", result.is_full_wave_inversion)?;
    out.set_item(
        "uses_nonlinear_wave_propagation",
        result.uses_nonlinear_wave_propagation,
    )?;
    out.set_item("uses_rayleigh_plesset", result.uses_rayleigh_plesset)?;
    let metrics = PyDict::new(py);
    metrics.set_item("fwi", metric3d_dict(py, &result.fwi_metrics)?)?;
    metrics.set_item(
        "rayleigh_plesset_cavitation",
        metric3d_dict(py, &result.cavitation_metrics)?,
    )?;
    metrics.set_item("fusion", metric3d_dict(py, &result.fusion_metrics)?)?;
    out.set_item("metrics", metrics)?;
    Ok(out)
}

fn placement_dict<'py>(
    py: Python<'py>,
    metrics: &DevicePlacementMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("min_body_clearance_m", metrics.min_body_clearance_m)?;
    out.set_item("mean_body_clearance_m", metrics.mean_body_clearance_m)?;
    out.set_item("max_body_clearance_m", metrics.max_body_clearance_m)?;
    out.set_item(
        "skin_contact_to_nearest_aperture_m",
        metrics.skin_contact_to_nearest_aperture_m,
    )?;
    Ok(out)
}

fn point_axis(
    points: &[kwavers::clinical::therapy::theranostic_guidance::Point2],
    x_axis: bool,
) -> Array1<f64> {
    Array1::from(
        points
            .iter()
            .map(|point| if x_axis { point.x_m } else { point.y_m })
            .collect::<Vec<_>>(),
    )
}

fn points3_to_array(points: &[Point3]) -> Array2<f64> {
    Array2::from_shape_fn((points.len(), 3), |(row, col)| match col {
        0 => points[row].x_m,
        1 => points[row].y_m,
        _ => points[row].z_m,
    })
}

fn placement_points3_to_array(points: &[PlacementPoint3]) -> Array2<f64> {
    Array2::from_shape_fn((points.len(), 3), |(row, col)| match col {
        0 => points[row].x_m,
        1 => points[row].y_m,
        _ => points[row].z_m,
    })
}

fn placement_context_skin_gap(context: &PlacementContext) -> f64 {
    context
        .therapy_points_m
        .iter()
        .chain(context.imaging_points_m.iter())
        .map(|point| {
            ((point.x_m - context.skin_contact_m.x_m).powi(2)
                + (point.y_m - context.skin_contact_m.y_m).powi(2)
                + (point.z_m - context.skin_contact_m.z_m).powi(2))
            .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn metric_dict<'py>(
    py: Python<'py>,
    metrics: &ReconstructionMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("pearson", metrics.pearson)?;
    out.set_item("nrmse", metrics.nrmse)?;
    out.set_item("dice_equal_area", metrics.dice_equal_area)?;
    out.set_item("cnr", metrics.cnr)?;
    Ok(out)
}

fn metric3d_dict<'py>(
    py: Python<'py>,
    metrics: &VolumeReconstructionMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("dice_equal_area", metrics.dice_equal_area)?;
    out.set_item("cnr", metrics.cnr)?;
    out.set_item("nrmse", metrics.nrmse)?;
    Ok(out)
}

fn kwavers_to_py(err: kwavers::core::error::KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers theranostic inverse failed: {err}"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_theranostic_inverse_from_ritk, m)?)?;
    m.add_function(wrap_pyfunction!(run_theranostic_nonlinear_3d_from_ritk, m)?)?;
    m.add_function(wrap_pyfunction!(
        plan_brain_helmet_placement_from_ritk_ct,
        m
    )?)?;
    Ok(())
}
