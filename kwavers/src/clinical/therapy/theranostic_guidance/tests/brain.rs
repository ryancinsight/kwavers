use ndarray::{Array2, Array3};

use super::super::{
    placement_metrics, plan_brain_helmet_placement, prepare_brain_slice, run_theranostic_inverse,
    AnatomyKind, BrainTargetSelection, TheranosticInverseConfig,
};

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
    let prepared = prepare_brain_slice(ct, 0.002, 0, BrainTargetSelection::OrganCentroid).unwrap();
    let mut config = TheranosticInverseConfig::new(AnatomyKind::Brain);
    config.element_count = 64;
    config.receiver_offsets = vec![16, 32];
    config.frequencies_hz = vec![220_000.0];
    config.iterations = 6;
    let result = run_theranostic_inverse(prepared, &config).unwrap();
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
    assert_eq!(result.measurements, result.encoded_measurements);
    assert_eq!(
        result.inverse_encoding_rows_per_code,
        config.inverse_encoding_rows_per_code
    );
    assert!(result.unencoded_measurements > result.encoded_measurements);
    assert!(result.operator_storage_values < result.dense_operator_values);
    let exposure_peak = result.exposure.iter().copied().fold(0.0, f64::max);
    assert!(
        (exposure_peak - config.source_pressure_pa).abs() <= config.source_pressure_pa * 1.0e-12,
        "exposure peak={exposure_peak}, expected pressure={}",
        config.source_pressure_pa
    );
}

#[test]
fn brain_slice_resampled_index_target_controls_focus_mask() {
    let mut ct = Array2::<f64>::from_elem((40, 40), -900.0);
    for x in 0..40 {
        for y in 0..40 {
            let head =
                ((x as f64 - 20.0) / 16.0).powi(2) + ((y as f64 - 20.0) / 15.0).powi(2) <= 1.0;
            if head {
                ct[[x, y]] = 40.0;
            }
        }
    }
    let requested = [25.0, 16.0];
    let prepared = prepare_brain_slice(
        ct,
        0.001,
        0,
        BrainTargetSelection::ResampledIndex(requested),
    )
    .unwrap();
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut count = 0.0;
    for ((ix, iy), active) in prepared.target_mask.indexed_iter() {
        if *active {
            sx += ix as f64;
            sy += iy as f64;
            count += 1.0;
        }
    }
    let centroid = [sx / count, sy / count];

    assert!(count > 64.0);
    assert!(
        (centroid[0] - requested[0]).abs() <= 0.25,
        "target centroid x={} must match requested x={}",
        centroid[0],
        requested[0]
    );
    assert!(
        (centroid[1] - requested[1]).abs() <= 0.25,
        "target centroid y={} must match requested y={}",
        centroid[1],
        requested[1]
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
        plan_brain_helmet_placement(&ct, [1.0, 1.0, 2.0], 128, 2, -300.0, 300.0, None, None)
            .unwrap();
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
    let min_unit_z = min_element_z / placement.helmet_radius_m;
    let max_unit_z = max_element_z / placement.helmet_radius_m;

    assert_eq!(placement.therapy_elements_m.len(), 128);
    assert!(
        min_element_z > -0.060,
        "helmet cap must not extend into inferior neck-like coverage: min_z={min_element_z}"
    );
    assert!(
        max_element_z > 0.090,
        "helmet cap must cover the superior calvarium: max_z={max_element_z}"
    );
    assert!(
        (min_unit_z + 0.28).abs() < 0.02,
        "focused bowl lower polar bound changed: min_unit_z={min_unit_z}"
    );
    assert!(
        (max_unit_z - 0.98).abs() < 0.02,
        "focused bowl upper polar bound changed: max_unit_z={max_unit_z}"
    );
    assert!(placement.intersection_fraction > 0.0);
}
