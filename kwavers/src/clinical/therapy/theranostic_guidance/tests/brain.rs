use ndarray::{Array2, Array3};

use super::super::{
    placement_metrics, plan_transcranial_focused_bowl_placement, prepare_brain_slice,
    run_theranostic_inverse, AnatomyKind, BrainTargetSelection, TheranosticInverseConfig,
};

#[test]
fn brain_focused_bowl_layout_uses_requested_element_count() {
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
    assert!(result.layout.model_name.contains("focused_bowl"));
    assert!(
        contact_radius > focus_radius,
        "focused bowl contact point must lie on the head boundary, focus radius={focus_radius}, contact radius={contact_radius}"
    );
    assert!(
        placement.skin_contact_to_nearest_aperture_m >= 0.010,
        "focused bowl aperture must remain outside the CT-derived head support"
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
fn brain_focused_bowl_3d_uses_calvarium_cap_not_inferior_hemisphere() {
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
    // Use the canonical calvarium-cap bounds: [0.22, 1.18] rad from vertex.
    // cos(0.22) ≈ 0.9759  (upper axis-projection limit — near-vertex cutoff)
    // cos(1.18) ≈ 0.3817  (lower axis-projection limit — calvarium boundary)
    let cap_min = 0.22_f64;
    let cap_max = 1.18_f64;
    let placement = plan_transcranial_focused_bowl_placement(
        &ct,
        [1.0, 1.0, 2.0],
        128,
        2,
        -300.0,
        300.0,
        None,
        None,
        Some(cap_min),
        Some(cap_max),
    )
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
    let min_unit_z = min_element_z / placement.bowl_radius_m;
    let max_unit_z = max_element_z / placement.bowl_radius_m;

    // cos(cap_max) ≈ 0.382 — all elements must be ABOVE the equatorial plane.
    // This verifies the calvarium-only constraint: no neck or jaw coverage.
    let expected_min_unit_z = cap_max.cos(); // ≈ 0.3817
    let expected_max_unit_z = cap_min.cos(); // ≈ 0.9759
    assert_eq!(placement.therapy_elements_m.len(), 128);
    assert!(
        min_element_z > 0.0,
        "focused bowl cap must not extend below equatorial plane (calvarium constraint): min_z={min_element_z}"
    );
    assert!(
        max_element_z > 0.090,
        "focused bowl cap must cover the superior calvarium: max_z={max_element_z}"
    );
    assert!(
        (min_unit_z - expected_min_unit_z).abs() < 0.04,
        "focused bowl lower polar bound (cos(cap_max)={expected_min_unit_z:.4}) \
         does not match actual min_unit_z={min_unit_z:.4}"
    );
    assert!(
        (max_unit_z - expected_max_unit_z).abs() < 0.04,
        "focused bowl upper polar bound (cos(cap_min)={expected_max_unit_z:.4}) \
         does not match actual max_unit_z={max_unit_z:.4}"
    );
    assert!(placement.intersection_fraction > 0.0);
}

#[test]
fn brain_focused_bowl_3d_rejects_invalid_configured_polar_bound() {
    let mut ct = Array3::<f64>::from_elem((24, 24, 14), -1000.0);
    for x in 0..24 {
        for y in 0..24 {
            for z in 0..14 {
                let r = ((x as f64 - 12.0) / 9.0).powi(2)
                    + ((y as f64 - 12.0) / 8.0).powi(2)
                    + ((z as f64 - 5.0) / 7.0).powi(2);
                if r <= 1.0 {
                    ct[[x, y, z]] = 40.0;
                }
                if (0.72..=1.0).contains(&r) {
                    ct[[x, y, z]] = 700.0;
                }
            }
        }
    }

    let error = plan_transcranial_focused_bowl_placement(
        &ct,
        [1.0, 1.0, 2.0],
        64,
        2,
        -300.0,
        300.0,
        None,
        None,
        Some(f64::NAN),
        Some(1.18),
    )
    .unwrap_err();

    assert!(
        format!("{error:?}").contains("cap_min_polar_rad"),
        "expected cap_min_polar_rad validation, got {error:?}"
    );
}
