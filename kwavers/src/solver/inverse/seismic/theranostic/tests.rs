use ndarray::{Array2, Array3};

use super::operator::active_grid;
use super::{
    build_abdominal_placement_context, placement_metrics, plan_brain_helmet_placement,
    prepare_abdominal_slice, prepare_brain_slice, run_theranostic_fwi, AnatomyKind,
    PlacementPoint3, Point2, TheranosticFwiConfig,
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
fn active_grid_graph_laplacian_matches_four_neighbor_energy() {
    let mask = Array2::<bool>::from_elem((2, 2), true);
    let active = active_grid(&mask, 1.0);
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let mut laplacian = [0.0_f32; 4];
    active.graph_laplacian_into(&values, &mut laplacian);
    let graph_energy = values
        .iter()
        .zip(laplacian.iter())
        .map(|(value, lap)| value * lap)
        .sum::<f32>();

    assert_eq!(laplacian, [-3.0, -1.0, 1.0, 3.0]);
    assert_eq!(graph_energy, 10.0);
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

#[test]
fn active_grid_graph_laplacian_matches_edge_energy() {
    let mut mask = Array2::<bool>::from_elem((3, 3), false);
    mask[[1, 0]] = true;
    mask[[1, 1]] = true;
    mask[[1, 2]] = true;
    mask[[2, 1]] = true;
    let active = active_grid(&mask, 1.0);
    let values = vec![1.0_f32, 2.0, 4.0, 8.0];
    let mut laplacian = vec![0.0_f32; active.len()];
    active.graph_laplacian_into(&values, &mut laplacian);
    let energy = values
        .iter()
        .zip(laplacian.iter())
        .map(|(value, lap)| f64::from(*value) * f64::from(*lap))
        .sum::<f64>();

    assert_eq!(laplacian, vec![-1.0, -7.0, 2.0, 6.0]);
    assert_eq!(energy, 41.0);
}

fn distance_2d(a: Point2, b: Point2) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}

fn skin_normal_projection_2d(point: Point2, skin: Point2, focus: Point2) -> f64 {
    let depth = distance_2d(skin, focus);
    let normal_x = (focus.x_m - skin.x_m) / depth;
    let normal_y = (focus.y_m - skin.y_m) / depth;
    (point.x_m - skin.x_m) * normal_x + (point.y_m - skin.y_m) * normal_y
}

fn skin_normal_projection_3d(
    point: PlacementPoint3,
    skin: PlacementPoint3,
    focus: PlacementPoint3,
) -> f64 {
    let dx = focus.x_m - skin.x_m;
    let dy = focus.y_m - skin.y_m;
    let depth = dx.hypot(dy);
    let normal_x = dx / depth;
    let normal_y = dy / depth;
    (point.x_m - skin.x_m) * normal_x + (point.y_m - skin.y_m) * normal_y
}
