//! Serialization of `TheranosticInverseResult` into a Python dict, plus
//! private geometry helpers used by `run_theranostic_inverse_from_ritk`.

#![cfg_attr(test, expect(clippy::unwrap_used, reason = "ratchet KWAVERS-UNWRAP-1"))]

use kwavers_therapy::therapy::theranostic_guidance::{
    placement_metrics, target_index_from_mask_fraction_3d, PlacementContext,
    TheranosticInverseConfig, TheranosticInverseResult, THERANOSTIC_OPERATOR_MODEL,
    TRANSMIT_SCHEDULE_MODEL,
};
use leto::Array3;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::super::helpers::{
    metric_dict, placement_context_skin_gap, placement_dict, point_axis, points3_to_array,
};
use crate::array_utils::{leto2_to_pyarray2, vec_to_pyarray1};

pub(super) fn brain_target_index(
    ct_volume_hu: &Array3<f64>,
    fraction: [f64; 3],
) -> kwavers_core::error::KwaversResult<[usize; 3]> {
    let brain = ct_volume_hu.mapv(|hu| hu > -300.0 && hu < 300.0);
    if brain.iter().any(|active| *active) {
        target_index_from_mask_fraction_3d(&brain, fraction)
    } else {
        let body = ct_volume_hu.mapv(|hu| hu > -300.0);
        target_index_from_mask_fraction_3d(&body, fraction)
    }
}

pub(super) fn resampled_crop_index_xy(
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

pub(super) fn result_to_dict<'py>(
    py: Python<'py>,
    result: TheranosticInverseResult,
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
    out.set_item("ct_hu", leto2_to_pyarray2(py, prepared.ct_hu)?)?;
    out.set_item("label", leto2_to_pyarray2(py, prepared.label)?)?;
    out.set_item(
        "sound_speed_m_s",
        leto2_to_pyarray2(py, prepared.sound_speed_m_s)?,
    )?;
    out.set_item(
        "attenuation_np_per_m_mhz",
        leto2_to_pyarray2(py, prepared.attenuation_np_per_m_mhz)?,
    )?;
    out.set_item("body_mask", leto2_to_pyarray2(py, prepared.body_mask)?)?;
    out.set_item("organ_mask", leto2_to_pyarray2(py, prepared.organ_mask)?)?;
    out.set_item("target_mask", leto2_to_pyarray2(py, prepared.target_mask)?)?;
    out.set_item("exposure", leto2_to_pyarray2(py, result.exposure)?)?;
    out.set_item(
        "exposure_raw_peak_pressure",
        leto2_to_pyarray2(py, result.exposure_raw_peak_pressure)?,
    )?;
    out.set_item(
        "lesion_target",
        leto2_to_pyarray2(py, result.lesion_target)?,
    )?;
    out.set_item(
        "anatomy_reconstruction",
        leto2_to_pyarray2(py, result.anatomy_reconstruction)?,
    )?;
    out.set_item(
        "active_lesion_reconstruction",
        leto2_to_pyarray2(py, result.active_lesion_reconstruction)?,
    )?;
    out.set_item(
        "waveform_rtm_reconstruction",
        leto2_to_pyarray2(py, result.waveform_rtm_reconstruction)?,
    )?;
    out.set_item(
        "elastic_shear_reconstruction",
        leto2_to_pyarray2(py, result.elastic_shear_reconstruction)?,
    )?;
    out.set_item(
        "subharmonic_reconstruction",
        leto2_to_pyarray2(py, result.subharmonic_reconstruction)?,
    )?;
    out.set_item(
        "harmonic_reconstruction",
        leto2_to_pyarray2(py, result.harmonic_reconstruction)?,
    )?;
    out.set_item(
        "ultraharmonic_reconstruction",
        leto2_to_pyarray2(py, result.ultraharmonic_reconstruction)?,
    )?;
    out.set_item(
        "fused_reconstruction",
        leto2_to_pyarray2(py, result.fused_reconstruction)?,
    )?;
    out.set_item(
        "therapy_x_m",
        vec_to_pyarray1(py, point_axis(&layout.therapy_elements, true)),
    )?;
    out.set_item(
        "therapy_y_m",
        vec_to_pyarray1(py, point_axis(&layout.therapy_elements, false)),
    )?;
    out.set_item(
        "imaging_x_m",
        vec_to_pyarray1(py, point_axis(&layout.imaging_receivers, true)),
    )?;
    out.set_item(
        "imaging_y_m",
        vec_to_pyarray1(py, point_axis(&layout.imaging_receivers, false)),
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
    out.set_item(
        "placement_ct_hu",
        leto2_to_pyarray2(py, placement_context.ct_hu)?,
    )?;
    out.set_item(
        "placement_body_mask",
        leto2_to_pyarray2(py, placement_context.body_mask)?,
    )?;
    out.set_item(
        "placement_target_mask",
        leto2_to_pyarray2(py, placement_context.target_mask)?,
    )?;
    out.set_item("placement_spacing_m", placement_spacing_m)?;
    out.set_item("placement_slice_index", placement_context.slice_index)?;
    out.set_item(
        "placement_therapy_points_m",
        placement_therapy_points.to_pyarray(py),
    )?;
    out.set_item(
        "placement_imaging_points_m",
        placement_imaging_points.to_pyarray(py),
    )?;
    out.set_item(
        "placement_body_surface_points_m",
        placement_body_surface_points.to_pyarray(py),
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
    // The elastic-shear channel runs iterative nonlinear ElasticPSTD FWI with
    // line search; this is exposed as its own flag so callers can distinguish
    // "any FWI happens in this result" from "the acoustic inverse is FWI"
    // (the latter is false — see [`THERANOSTIC_FULL_WAVE_INVERSION`]).
    out.set_item(
        "iterative_elastic_fwi",
        kwavers_therapy::therapy::theranostic_guidance::THERANOSTIC_ITERATIVE_ELASTIC_FWI,
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
        vec_to_pyarray1(py, result.objective_history),
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

#[cfg(test)]
mod tests {
    use super::{brain_target_index, resampled_crop_index_xy};

    #[test]
    fn brain_target_index_resolves_inside_brain_support() {
        // 5x5x5 volume: air (-1000 HU) outside, soft tissue (0 HU) block at 1..=3.
        let mut ct = leto::Array3::from_elem((5, 5, 5), -1000.0);
        for ix in 1..=3 {
            for iy in 1..=3 {
                for iz in 1..=3 {
                    ct[[ix, iy, iz]] = 0.0;
                }
            }
        }

        let index = brain_target_index(&ct, [0.5, 0.5, 0.5]).unwrap();
        assert_eq!(index, [2, 2, 2]);
    }

    #[test]
    fn brain_target_index_falls_back_to_body_when_brain_empty() {
        // Every voxel is 500 HU: no voxel lies in (-300, 300), so the body
        // mask (hu > -300) is used and the full support resolves to the center.
        let ct = leto::Array3::from_elem((7, 7, 7), 500.0);

        let index = brain_target_index(&ct, [0.5, 0.5, 0.5]).unwrap();
        assert_eq!(index, [3, 3, 3]);
    }

    #[test]
    fn brain_target_index_rejects_volume_without_body_voxels() {
        // Nothing exceeds -300 HU, so both the brain and body masks are empty
        // and the target resolution fails instead of guessing.
        let ct = leto::Array3::from_elem((4, 4, 4), -1000.0);

        assert!(brain_target_index(&ct, [0.5, 0.5, 0.5]).is_err());
    }

    #[test]
    fn resampled_crop_index_xy_maps_source_into_unit_grid() {
        // grid 9x9 → scale 8; crop [1,5] x [2,6] maps source (2,3) to (2.0, 2.0).
        let index = resampled_crop_index_xy([2, 3, 0], [1, 5, 2, 6], 9);
        assert_eq!(index, [2.0, 2.0]);
    }

    #[test]
    fn resampled_crop_index_xy_maps_crop_edges_to_grid_ends() {
        // Source on the crop edge maps to 0.0; source on the opposite edge
        // maps to the full scale (grid_size - 1).
        let index = resampled_crop_index_xy([1, 6, 0], [1, 5, 2, 6], 9);
        assert_eq!(index, [0.0, 8.0]);
    }

    #[test]
    fn resampled_crop_index_xy_guards_degenerate_crop_span() {
        // x1 == x0 would divide by zero; the .max(1.0) guard resolves the
        // degenerate axis through a unit span instead.
        let index = resampled_crop_index_xy([3, 3, 0], [3, 3, 0, 4], 9);
        assert_eq!(index, [0.0, 6.0]);
    }
}
