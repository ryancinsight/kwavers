use ndarray::Array3;

use super::super::{
    build_abdominal_placement_context, placement_metrics, prepare_abdominal_slice,
    run_theranostic_inverse, AnatomyKind, TheranosticInverseConfig, WaveformMisfit,
    THERANOSTIC_ELASTIC_SHEAR_MODEL, THERANOSTIC_INVERSE_MODEL_FAMILY, THERANOSTIC_WAVEFORM_MODEL,
};
use super::helpers::{
    connected_mask_components, distance_2d, nearest_mask_distance_m, skin_normal_projection_2d,
    skin_normal_projection_3d,
};

#[test]
fn abdominal_theranostic_inverse_recovers_lesion_support() {
    let mut ct = Array3::<f64>::from_elem((42, 42, 3), -900.0);
    let mut label = Array3::<i16>::zeros((42, 42, 3));
    for z in 0..3 {
        for x in 0..42 {
            for y in 0..42 {
                let body =
                    ((x as f64 - 21.0) / 18.0).powi(2) + ((y as f64 - 21.0) / 17.0).powi(2) <= 1.0;
                let organ =
                    ((x as f64 - 23.0) / 12.0).powi(2) + ((y as f64 - 22.0) / 11.0).powi(2) <= 1.0;
                let tumor =
                    ((x as f64 - 24.0) / 4.0).powi(2) + ((y as f64 - 22.0) / 3.5).powi(2) <= 1.0;
                if body {
                    ct[[x, y, z]] = -40.0;
                }
                if organ {
                    ct[[x, y, z]] = 70.0;
                    label[[x, y, z]] = 1;
                }
                if tumor {
                    ct[[x, y, z]] = 110.0;
                    label[[x, y, z]] = 2;
                }
            }
        }
    }
    let prepared =
        prepare_abdominal_slice(AnatomyKind::Kidney, &ct, &label, [1.4, 1.4, 3.0], 42).unwrap();
    let mut config = TheranosticInverseConfig::new(AnatomyKind::Kidney);
    config.element_count = 32;
    config.receiver_offsets = vec![8, 12];
    config.frequencies_hz = vec![260_000.0, 520_000.0];
    config.iterations = 10;
    let result = run_theranostic_inverse(prepared, &config).unwrap();
    let placement = placement_metrics(
        &result.layout,
        &result.prepared.body_mask,
        result.prepared.spacing_m,
    );

    assert_eq!(result.layout.therapy_elements.len(), 32);
    assert!(result.layout.model_name.contains("skin_coupled"));
    assert!(
        placement.skin_contact_to_nearest_aperture_m <= 0.004,
        "skin-contact aperture gap={}",
        placement.skin_contact_to_nearest_aperture_m
    );
    assert!(placement.min_body_clearance_m > 0.0);
    let exposure_peak = result.exposure.iter().copied().fold(0.0, f64::max);
    assert!(
        (exposure_peak - config.source_pressure_pa).abs() <= config.source_pressure_pa * 1.0e-12,
        "exposure peak={exposure_peak}, expected pressure={}",
        config.source_pressure_pa
    );
    assert!(
        result.active_metrics.dice_equal_area > 0.30,
        "active dice={}",
        result.active_metrics.dice_equal_area
    );
    assert!(
        result.subharmonic_metrics.dice_equal_area > 0.30,
        "subharmonic dice={}",
        result.subharmonic_metrics.dice_equal_area
    );
    assert!(result.measurements > 0);
    assert_eq!(result.measurements, result.encoded_measurements);
    assert_eq!(
        result.inverse_encoding_rows_per_code,
        config.inverse_encoding_rows_per_code
    );
    assert!(
        result.unencoded_measurements > result.encoded_measurements,
        "source encoding must reduce clinical inverse rows: encoded={}, unencoded={}",
        result.encoded_measurements,
        result.unencoded_measurements
    );
    assert_eq!(
        result.operator_backend,
        "matrix_free_finite_frequency_same_aperture"
    );
    assert!(result.operator_storage_values < result.dense_operator_values);
    assert_eq!(result.waveform.model_name, THERANOSTIC_WAVEFORM_MODEL);
    assert_eq!(
        result.waveform.misfit_name,
        WaveformMisfit::Charbonnier.label()
    );
    assert!(result.waveform.misfit_scale > 0.0);
    assert!(result.waveform.objective_value > 0.0);
    assert_eq!(
        result.inverse_model_family,
        THERANOSTIC_INVERSE_MODEL_FAMILY
    );
    assert_eq!(result.elastic_shear_model, THERANOSTIC_ELASTIC_SHEAR_MODEL);
    assert_eq!(
        result.elastic_shear_reconstruction.dim(),
        result.lesion_target.dim()
    );
    let elastic_peak = peak_index(
        &result.elastic_shear_reconstruction,
        &result.prepared.body_mask,
    );
    let lesion_peak = peak_index(&result.lesion_target, &result.prepared.body_mask);
    assert!(
        result.elastic_shear_metrics.dice_equal_area > 0.10,
        "elastic shear dice={}, elastic_peak={:?}, lesion_peak={:?}",
        result.elastic_shear_metrics.dice_equal_area,
        elastic_peak,
        lesion_peak
    );
    assert!(
        result.elastic_shear_metrics.cnr > 0.0,
        "elastic shear cnr={}",
        result.elastic_shear_metrics.cnr
    );
    assert_eq!(
        result.elastic_shear.model_name,
        THERANOSTIC_ELASTIC_SHEAR_MODEL
    );
    assert!(result.elastic_shear.receiver_count > 0);
    assert!(result.elastic_shear.time_steps >= 32);
    assert!(result.elastic_shear.dt_s > 0.0);
    assert!(result.elastic_shear.iteration_count > 0);
    assert!(result.elastic_shear.accepted_step_count > 0);
    assert!(result.elastic_shear.objective_history.len() >= 2);
    assert!(
        result.elastic_shear.objective_history.last().unwrap()
            < result.elastic_shear.objective_history.first().unwrap(),
        "elastic FWI objective history={:?}",
        result.elastic_shear.objective_history
    );
    assert!(result.elastic_shear.baseline_trace_energy > 0.0);
    assert!(result.elastic_shear.lesion_trace_energy > 0.0);
    assert!(result.elastic_shear.residual_trace_energy > 0.0);
    // The 2-D acoustic theranostic guidance pipeline is reduced-Born /
    // Tikhonov + one-pass adjoint RTM, not FWI. The flag must report this
    // honestly; full-waveform inversion is exercised by the nonlinear-3-D
    // Westervelt pipeline tests.
    assert!(!result.is_full_wave_inversion);
    assert!(!result.uses_nonlinear_wave_propagation);
    assert_eq!(
        result.waveform.receiver_count,
        result.layout.therapy_elements.len() + result.layout.imaging_receivers.len()
    );
    assert!(result.waveform.time_steps >= 96);
    assert!(result.waveform.dt_s > 0.0);
    assert!(result.waveform.observed_energy > 0.0);
    assert!(result.waveform.residual_energy > 0.0);
    assert!(
        result.waveform_metrics.cnr > 0.0,
        "waveform cnr={}",
        result.waveform_metrics.cnr
    );
}

fn peak_index(
    image: &ndarray::Array2<f64>,
    mask: &ndarray::Array2<bool>,
) -> Option<(usize, usize)> {
    image
        .indexed_iter()
        .filter(|(idx, _)| mask[*idx])
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

#[test]
fn abdominal_preprocessing_keeps_external_skin_between_target_and_aperture() {
    let mut ct = Array3::<f64>::from_elem((96, 96, 3), -950.0);
    let mut label = Array3::<i16>::zeros((96, 96, 3));
    for z in 0..3 {
        for x in 0..96 {
            for y in 0..96 {
                let body =
                    ((x as f64 - 48.0) / 39.0).powi(2) + ((y as f64 - 48.0) / 34.0).powi(2) <= 1.0;
                let organ =
                    ((x as f64 - 61.0) / 18.0).powi(2) + ((y as f64 - 48.0) / 14.0).powi(2) <= 1.0;
                let tumor =
                    ((x as f64 - 70.0) / 4.0).powi(2) + ((y as f64 - 48.0) / 3.0).powi(2) <= 1.0;
                if body {
                    ct[[x, y, z]] = -35.0;
                }
                if organ {
                    ct[[x, y, z]] = 65.0;
                    label[[x, y, z]] = 1;
                }
                if tumor {
                    ct[[x, y, z]] = 120.0;
                    label[[x, y, z]] = 2;
                }
            }
        }
    }

    let prepared =
        prepare_abdominal_slice(AnatomyKind::Liver, &ct, &label, [1.5, 1.5, 3.0], 40).unwrap();
    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.element_count = 32;
    config.receiver_offsets = vec![8];
    config.frequencies_hz = vec![500_000.0];
    config.iterations = 3;
    let result = run_theranostic_inverse(prepared, &config).unwrap();
    let target_depth_m = distance_2d(result.layout.skin_contact_m, result.layout.focus_m);
    let max_aperture_skin_projection_m = result
        .layout
        .therapy_elements
        .iter()
        .chain(result.layout.imaging_receivers.iter())
        .map(|point| {
            skin_normal_projection_2d(*point, result.layout.skin_contact_m, result.layout.focus_m)
        })
        .fold(f64::NEG_INFINITY, f64::max);

    assert!(
        target_depth_m > 0.020,
        "skin contact must come from the nearest external CT body boundary rather than the ROI crop edge, depth={target_depth_m}"
    );
    assert!(
        max_aperture_skin_projection_m < 0.0,
        "therapy and imaging aperture must remain outside the local skin tangent plane: projection={max_aperture_skin_projection_m}"
    );
}

#[test]
fn abdominal_preprocessing_selects_one_connected_treatment_component() {
    let mut ct = Array3::<f64>::from_elem((72, 72, 3), -950.0);
    let mut label = Array3::<i16>::zeros((72, 72, 3));
    for z in 0..3 {
        for x in 0..72 {
            for y in 0..72 {
                let body =
                    ((x as f64 - 36.0) / 31.0).powi(2) + ((y as f64 - 36.0) / 28.0).powi(2) <= 1.0;
                let organ =
                    ((x as f64 - 39.0) / 22.0).powi(2) + ((y as f64 - 36.0) / 19.0).powi(2) <= 1.0;
                let larger_tumor =
                    ((x as f64 - 52.0) / 5.0).powi(2) + ((y as f64 - 37.0) / 4.0).powi(2) <= 1.0;
                let smaller_tumor =
                    ((x as f64 - 28.0) / 2.6).powi(2) + ((y as f64 - 24.0) / 2.2).powi(2) <= 1.0;
                if body {
                    ct[[x, y, z]] = -35.0;
                }
                if organ {
                    ct[[x, y, z]] = 70.0;
                    label[[x, y, z]] = 1;
                }
                if larger_tumor || smaller_tumor {
                    ct[[x, y, z]] = 125.0;
                    label[[x, y, z]] = 2;
                }
            }
        }
    }

    let prepared =
        prepare_abdominal_slice(AnatomyKind::Liver, &ct, &label, [1.0, 1.0, 2.5], 64).unwrap();
    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.element_count = 32;
    config.receiver_offsets = vec![8];
    config.frequencies_hz = vec![500_000.0];
    config.iterations = 2;
    let result = run_theranostic_inverse(prepared.clone(), &config).unwrap();
    let focus_distance = nearest_mask_distance_m(
        &prepared.target_mask,
        prepared.spacing_m,
        result.layout.focus_m,
    );

    assert_eq!(
        connected_mask_components(&prepared.target_mask),
        1,
        "one Chapter 29 abdominal solve must represent one connected sonication target component"
    );
    assert!(
        focus_distance <= prepared.spacing_m,
        "single-focus layout must focus inside the selected connected target, distance={focus_distance}"
    );
}

#[test]
fn abdominal_placement_context_uses_uncropped_patient_slice() {
    let mut ct = Array3::<f64>::from_elem((96, 96, 3), -950.0);
    let mut label = Array3::<i16>::zeros((96, 96, 3));
    for z in 0..3 {
        for x in 0..96 {
            for y in 0..96 {
                let body =
                    ((x as f64 - 48.0) / 39.0).powi(2) + ((y as f64 - 48.0) / 34.0).powi(2) <= 1.0;
                let organ =
                    ((x as f64 - 61.0) / 18.0).powi(2) + ((y as f64 - 48.0) / 14.0).powi(2) <= 1.0;
                let tumor =
                    ((x as f64 - 70.0) / 4.0).powi(2) + ((y as f64 - 48.0) / 3.0).powi(2) <= 1.0;
                let internal_air =
                    ((x as f64 - 65.0) / 3.0).powi(2) + ((y as f64 - 48.0) / 4.0).powi(2) <= 1.0;
                if body {
                    ct[[x, y, z]] = -35.0;
                }
                if organ {
                    ct[[x, y, z]] = 65.0;
                    label[[x, y, z]] = 1;
                }
                if tumor {
                    ct[[x, y, z]] = 120.0;
                    label[[x, y, z]] = 2;
                }
                if internal_air {
                    ct[[x, y, z]] = -950.0;
                    label[[x, y, z]] = 0;
                }
            }
        }
    }
    let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
    config.element_count = 32;
    let context = build_abdominal_placement_context(
        AnatomyKind::Liver,
        &ct,
        &label,
        [1.5, 1.5, 3.0],
        &config,
    )
    .unwrap();
    let max_aperture_skin_projection_m = context
        .therapy_points_m
        .iter()
        .chain(context.imaging_points_m.iter())
        .map(|point| skin_normal_projection_3d(*point, context.skin_contact_m, context.focus_m))
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(context.ct_hu.dim(), (96, 96));
    assert_eq!(context.therapy_points_m.len(), 32);
    assert_eq!(context.imaging_points_m.len(), 64);
    assert!(
        context.skin_contact_m.x_m > context.focus_m.x_m,
        "right-sided target must use the nearest external right skin boundary rather than a fixed left-edge contact"
    );
    assert!(
        max_aperture_skin_projection_m < 0.0,
        "abdominal aperture must remain outside the local skin tangent plane: projection={max_aperture_skin_projection_m}"
    );
}
