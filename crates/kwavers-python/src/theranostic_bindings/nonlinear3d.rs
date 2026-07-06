//! `run_theranostic_nonlinear_3d_from_ritk` pyfunction and result serialization.

use kwavers_therapy::therapy::theranostic_guidance::{
    run_theranostic_nonlinear_3d, AnatomyKind, Nonlinear3dConfig,
};
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::Path;

use super::helpers::{kwavers_to_py, labels_from_volume, metric3d_dict, points3_to_array};
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
    target_fraction_xyz = None,
    frequency_hz = None,
    source_pressure_pa = None,
    cycles = 3.0,
    treatment_window_radius_m = 0.04,
    min_points_per_wavelength = 6.0,
    lesion_delta_c_m_s = -35.0,
    lesion_delta_beta = 0.85,
    sound_speed_regularization = 2.0e-3,
    nonlinearity_regularization = 1.0e-3,
    gradient_smoothing_steps = 2,
    bubble_radius_m = 2.0e-6,
    bubble_time_steps_per_period = 96,
    inertial_mi_threshold = 1.9,
    cavitation_iterations = 24,
    cavitation_regularization = 1.0e-4
))]
#[allow(clippy::too_many_arguments)]
pub fn run_theranostic_nonlinear_3d_from_ritk<'py>(
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
    target_fraction_xyz: Option<(f64, f64, f64)>,
    frequency_hz: Option<f64>,
    source_pressure_pa: Option<f64>,
    cycles: f64,
    treatment_window_radius_m: f64,
    min_points_per_wavelength: f64,
    lesion_delta_c_m_s: f64,
    lesion_delta_beta: f64,
    sound_speed_regularization: f64,
    nonlinearity_regularization: f64,
    gradient_smoothing_steps: usize,
    bubble_radius_m: f64,
    bubble_time_steps_per_period: usize,
    inertial_mi_threshold: f64,
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
    config.treatment_window_radius_m = treatment_window_radius_m;
    config.min_points_per_wavelength = min_points_per_wavelength;
    config.lesion_delta_c_m_s = lesion_delta_c_m_s;
    config.lesion_delta_beta = lesion_delta_beta;
    config.sound_speed_regularization = sound_speed_regularization;
    config.nonlinearity_regularization = nonlinearity_regularization;
    config.gradient_smoothing_steps = gradient_smoothing_steps;
    config.bubble_radius_m = bubble_radius_m;
    config.bubble_time_steps_per_period = bubble_time_steps_per_period;
    config.inertial_mi_threshold = inertial_mi_threshold;
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
    // `ct` and `labels` are moved into `run_theranostic_nonlinear_3d` so the
    // full-resolution CT arrays are freed before the Westervelt FWI loop.
    let result = py
        .detach(|| {
            run_theranostic_nonlinear_3d(
                anatomy,
                ct,
                labels,
                spacing_mm,
                &config,
                target_fraction_xyz.map(|(x, y, z)| [x, y, z]),
            )
        })
        .map_err(kwavers_to_py)?;
    nonlinear3d_result_to_dict(py, result, &config, target_fraction_xyz)
}

pub(super) fn nonlinear3d_result_to_dict<'py>(
    py: Python<'py>,
    result: kwavers_therapy::therapy::theranostic_guidance::Nonlinear3dResult,
    config: &Nonlinear3dConfig,
    target_fraction_xyz: Option<(f64, f64, f64)>,
) -> PyResult<Bound<'py, PyDict>> {
    use ndarray::Array1;
    let out = PyDict::new(py);
    out.set_item("anatomy", config.anatomy.label())?;
    out.set_item("ct_hu", result.ct_hu.to_pyarray(py))?;
    out.set_item("label", result.label.to_pyarray(py))?;
    out.set_item("body_mask", result.body_mask.to_pyarray(py))?;
    out.set_item("target_mask", result.target_mask.to_pyarray(py))?;
    out.set_item("inversion_mask", result.inversion_mask.to_pyarray(py))?;
    out.set_item(
        "background_sound_speed_m_s",
        result.background_sound_speed_m_s.to_pyarray(py),
    )?;
    out.set_item(
        "true_sound_speed_m_s",
        result.true_sound_speed_m_s.to_pyarray(py),
    )?;
    out.set_item(
        "reconstructed_sound_speed_m_s",
        result.reconstructed_sound_speed_m_s.to_pyarray(py),
    )?;
    out.set_item(
        "reconstructed_delta_c_m_s",
        result.reconstructed_delta_c_m_s.to_pyarray(py),
    )?;
    out.set_item("background_beta", result.background_beta.to_pyarray(py))?;
    out.set_item("true_beta", result.true_beta.to_pyarray(py))?;
    out.set_item(
        "reconstructed_beta",
        result.reconstructed_beta.to_pyarray(py),
    )?;
    out.set_item(
        "reconstructed_delta_beta",
        result.reconstructed_delta_beta.to_pyarray(py),
    )?;
    out.set_item(
        "multiparameter_fwi_score",
        result.multiparameter_fwi_score.to_pyarray(py),
    )?;
    out.set_item(
        "nonlinear_fusion_score",
        result.nonlinear_fusion_score.to_pyarray(py),
    )?;
    out.set_item(
        "westervelt_peak_pressure_pa",
        result.westervelt_peak_pressure_pa.to_pyarray(py),
    )?;
    out.set_item(
        "cavitation_source_density",
        result.cavitation_source_density.to_pyarray(py),
    )?;
    out.set_item(
        "reconstructed_cavitation_density",
        result.reconstructed_cavitation_density.to_pyarray(py),
    )?;
    out.set_item(
        "fwi_objective_history",
        Array1::from(result.fwi_objective_history).to_pyarray(py),
    )?;
    let fwi_iteration_diagnostics = PyList::empty(py);
    for diagnostic in &result.fwi_iteration_diagnostics {
        let item = PyDict::new(py);
        item.set_item("objective_before", diagnostic.objective_before)?;
        item.set_item("objective_after", diagnostic.objective_after)?;
        item.set_item("gradient_speed_linf", diagnostic.gradient_speed_linf)?;
        item.set_item("gradient_beta_linf", diagnostic.gradient_beta_linf)?;
        item.set_item("gradient_speed_l2", diagnostic.gradient_speed_l2)?;
        item.set_item("gradient_beta_l2", diagnostic.gradient_beta_l2)?;
        item.set_item("accepted_scale", diagnostic.accepted_scale)?;
        item.set_item("accepted_block", diagnostic.accepted_block)?;
        fwi_iteration_diagnostics.append(item)?;
    }
    out.set_item("fwi_iteration_diagnostics", fwi_iteration_diagnostics)?;
    out.set_item(
        "cavitation_objective_history",
        Array1::from(result.cavitation_objective_history).to_pyarray(py),
    )?;
    out.set_item(
        "therapy_points_m",
        points3_to_array(&result.therapy_points_m).to_pyarray(py),
    )?;
    out.set_item(
        "receiver_points_m",
        points3_to_array(&result.receiver_points_m).to_pyarray(py),
    )?;
    out.set_item("spacing_m", result.spacing_m)?;
    out.set_item("dt_s", result.dt_s)?;
    out.set_item("time_steps", result.time_steps)?;
    out.set_item("source_scale", result.source_scale)?;
    let source_plan_metrics = PyDict::new(py);
    source_plan_metrics.set_item(
        "source_support_min",
        result.source_plan_metrics.source_support_min,
    )?;
    source_plan_metrics.set_item(
        "source_support_mean",
        result.source_plan_metrics.source_support_mean,
    )?;
    source_plan_metrics.set_item(
        "source_support_max",
        result.source_plan_metrics.source_support_max,
    )?;
    source_plan_metrics.set_item(
        "focused_delay_min_s",
        result.source_plan_metrics.focused_delay_min_s,
    )?;
    source_plan_metrics.set_item(
        "focused_delay_max_s",
        result.source_plan_metrics.focused_delay_max_s,
    )?;
    source_plan_metrics.set_item(
        "focused_delay_span_s",
        result.source_plan_metrics.focused_delay_span_s,
    )?;
    out.set_item("source_plan_metrics", source_plan_metrics)?;
    let steering_metrics = PyDict::new(py);
    steering_metrics.set_item(
        "nominal_focus_index",
        result
            .electronic_steering_metrics
            .nominal_focus_index
            .to_vec(),
    )?;
    steering_metrics.set_item(
        "calibration_hotspot_index",
        result
            .electronic_steering_metrics
            .calibration_hotspot_index
            .to_vec(),
    )?;
    steering_metrics.set_item(
        "steering_focus_index",
        result
            .electronic_steering_metrics
            .steering_focus_index
            .to_vec(),
    )?;
    steering_metrics.set_item(
        "correction_grid_cells",
        result
            .electronic_steering_metrics
            .correction_grid_cells
            .to_vec(),
    )?;
    steering_metrics.set_item(
        "calibration_hotspot_distance_grid_cells",
        result
            .electronic_steering_metrics
            .calibration_hotspot_distance_grid_cells,
    )?;
    steering_metrics.set_item(
        "steering_applied",
        result.electronic_steering_metrics.steering_applied,
    )?;
    out.set_item("electronic_steering_metrics", steering_metrics)?;
    out.set_item("active_voxels", result.active_voxels)?;
    out.set_item("grid_size", config.grid_size)?;
    out.set_item("requested_element_count", config.element_count)?;
    out.set_item("requested_receiver_count", config.receiver_count)?;
    out.set_item("element_count", result.therapy_points_m.len())?;
    out.set_item("receiver_count", result.receiver_points_m.len())?;
    out.set_item("source_encoding_count", config.source_encoding_count)?;
    out.set_item("source_dimensions", result.source_dimensions.to_vec())?;
    out.set_item("source_spacing_m", result.source_spacing_m.to_vec())?;
    out.set_item("crop_bounds_index", result.crop_bounds_index.to_vec())?;
    out.set_item(
        "treatment_window_radius_m",
        result.treatment_window_radius_m,
    )?;
    out.set_item("wavelength_min_m", result.wavelength_min_m)?;
    out.set_item(
        "points_per_wavelength_min",
        result.points_per_wavelength_min,
    )?;
    out.set_item(
        "min_points_per_wavelength",
        config.min_points_per_wavelength,
    )?;
    out.set_item("resolution_meets_min_ppw", result.resolution_meets_min_ppw)?;
    out.set_item(
        "checkpoint_interval_steps",
        config.checkpoint_interval_steps,
    )?;
    out.set_item("frequency_hz", config.frequency_hz)?;
    out.set_item("source_pressure_pa", config.source_pressure_pa)?;
    if let Some((x, y, z)) = target_fraction_xyz {
        out.set_item("target_fraction_xyz", (x, y, z))?;
    }
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
    out.set_item(
        "bubble_time_steps_per_period",
        config.bubble_time_steps_per_period,
    )?;
    out.set_item("inertial_mi_threshold", config.inertial_mi_threshold)?;
    out.set_item("cavitation_iterations", config.cavitation_iterations)?;
    out.set_item(
        "cavitation_regularization",
        config.cavitation_regularization,
    )?;
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

