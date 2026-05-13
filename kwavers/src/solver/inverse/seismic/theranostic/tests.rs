use ndarray::{Array2, Array3};

use super::{
    build_abdominal_placement_context, placement_metrics, plan_brain_helmet_placement,
    prepare_abdominal_slice, prepare_brain_slice, run_theranostic_fwi, AnatomyKind,
    TheranosticFwiConfig,
};

#[test]
fn abdominal_theranostic_fwi_recovers_lesion_support() {
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
    let mut config = TheranosticFwiConfig::new(AnatomyKind::Kidney);
    config.element_count = 32;
    config.receiver_offsets = vec![8, 12];
    config.frequencies_hz = vec![260_000.0, 520_000.0];
    config.iterations = 10;
    let result = run_theranostic_fwi(prepared, &config).unwrap();
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
    let mut config = TheranosticFwiConfig::new(AnatomyKind::Liver);
    config.element_count = 32;
    config.receiver_offsets = vec![8];
    config.frequencies_hz = vec![500_000.0];
    config.iterations = 3;
    let result = run_theranostic_fwi(prepared, &config).unwrap();
    let nearest_imaging_x = result
        .layout
        .imaging_receivers
        .iter()
        .map(|point| point.x_m)
        .fold(f64::INFINITY, f64::min);
    let target_depth_m = result.layout.focus_m.x_m - result.layout.skin_contact_m.x_m;

    assert!(
        target_depth_m > 0.060,
        "skin contact must come from the external CT body boundary, depth={target_depth_m}"
    );
    assert!(
        nearest_imaging_x < result.layout.skin_contact_m.x_m,
        "central imaging receiver must remain outside the skin contact point"
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
    let mut config = TheranosticFwiConfig::new(AnatomyKind::Liver);
    config.element_count = 32;
    let context = build_abdominal_placement_context(
        AnatomyKind::Liver,
        &ct,
        &label,
        [1.5, 1.5, 3.0],
        &config,
    )
    .unwrap();
    let aperture_max_x = context
        .therapy_points_m
        .iter()
        .chain(context.imaging_points_m.iter())
        .map(|point| point.x_m)
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(context.ct_hu.dim(), (96, 96));
    assert_eq!(context.therapy_points_m.len(), 32);
    assert_eq!(context.imaging_points_m.len(), 64);
    assert!(
        aperture_max_x < context.skin_contact_m.x_m,
        "abdominal aperture must remain outside patient skin: aperture_max_x={aperture_max_x}, skin_x={}",
        context.skin_contact_m.x_m
    );
}

#[test]
fn brain_helmet_layout_uses_requested_element_count() {
    let mut ct = Array2::<f64>::from_elem((36, 36), -900.0);
    for x in 0..36 {
        for y in 0..36 {
            let head =
                ((x as f64 - 18.0) / 14.0).powi(2) + ((y as f64 - 18.0) / 13.0).powi(2) <= 1.0;
            let skull = ((x as f64 - 18.0) / 14.0).powi(2) + ((y as f64 - 18.0) / 13.0).powi(2)
                >= 0.82
                && head;
            if head {
                ct[[x, y]] = 40.0;
            }
            if skull {
                ct[[x, y]] = 700.0;
            }
        }
    }
    let prepared = prepare_brain_slice(ct, 0.002, 0).unwrap();
    let mut config = TheranosticFwiConfig::new(AnatomyKind::Brain);
    config.element_count = 64;
    config.receiver_offsets = vec![16, 32];
    config.frequencies_hz = vec![220_000.0];
    config.iterations = 6;
    let result = run_theranostic_fwi(prepared, &config).unwrap();
    let placement = placement_metrics(
        &result.layout,
        &result.prepared.body_mask,
        result.prepared.spacing_m,
    );
    let focus_radius = result.layout.focus_m.x_m.hypot(result.layout.focus_m.y_m);
    let contact_radius = result
        .layout
        .skin_contact_m
        .x_m
        .hypot(result.layout.skin_contact_m.y_m);

    assert_eq!(result.layout.therapy_elements.len(), 64);
    assert!(result.layout.model_name.contains("helmet"));
    assert!(
        contact_radius > focus_radius,
        "helmet contact point must lie on the head boundary, focus radius={focus_radius}, contact radius={contact_radius}"
    );
    assert!(
        placement.skin_contact_to_nearest_aperture_m >= 0.010,
        "helmet aperture must remain outside the CT-derived head support"
    );
    assert!(placement.min_body_clearance_m >= 0.010);
    assert!(result.active_voxels > 16);
    assert!(result.active_metrics.cnr > 0.0);
    let exposure_peak = result.exposure.iter().copied().fold(0.0, f64::max);
    assert!(
        (exposure_peak - config.source_pressure_pa).abs() <= config.source_pressure_pa * 1.0e-12,
        "exposure peak={exposure_peak}, expected pressure={}",
        config.source_pressure_pa
    );
}

#[test]
fn brain_helmet_3d_uses_calvarium_cap_not_inferior_hemisphere() {
    let mut ct = Array3::<f64>::from_elem((36, 36, 18), -1000.0);
    for x in 0..36 {
        for y in 0..36 {
            for z in 0..18 {
                let r = ((x as f64 - 18.0) / 13.0).powi(2)
                    + ((y as f64 - 18.0) / 12.0).powi(2)
                    + ((z as f64 - 6.0) / 9.0).powi(2);
                if r <= 1.0 {
                    ct[[x, y, z]] = 40.0;
                }
                if (0.72..=1.0).contains(&r) {
                    ct[[x, y, z]] = 700.0;
                }
            }
        }
    }
    let placement =
        plan_brain_helmet_placement(&ct, [1.0, 1.0, 2.0], 128, 2, -300.0, 300.0).unwrap();
    let min_element_z = placement
        .therapy_elements_m
        .iter()
        .map(|point| point.z_m)
        .fold(f64::INFINITY, f64::min);
    let max_element_z = placement
        .therapy_elements_m
        .iter()
        .map(|point| point.z_m)
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(placement.therapy_elements_m.len(), 128);
    assert!(
        min_element_z > -0.060,
        "helmet cap must not extend into inferior neck-like coverage: min_z={min_element_z}"
    );
    assert!(
        max_element_z > 0.090,
        "helmet cap must cover the superior calvarium: max_z={max_element_z}"
    );
    assert!(placement.intersection_fraction > 0.0);
}
