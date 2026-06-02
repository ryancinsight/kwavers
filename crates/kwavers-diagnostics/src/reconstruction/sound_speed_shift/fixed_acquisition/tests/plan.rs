use std::f64::consts::TAU;

use super::*;

#[test]
fn fixed_plan_reconstruction_matches_direct_reconstruction() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(20.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 32,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);
    let direct = reconstruct_sound_speed_shift(&measured, &mask, config).unwrap();
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let mut workspace = SoundSpeedShiftWorkspace::new();

    let planned = plan
        .reconstruct_with_workspace(&predicted, &mut workspace)
        .unwrap();

    assert_eq!(plan.rows_available(), 3);
    assert_eq!(plan.rows_used(), 3);
    assert_eq!(plan.active_voxels(), 9);
    assert!(plan.stored_weight_count() > 0);
    assert_image_close(
        &planned.sound_speed_shift_m_s,
        &direct.sound_speed_shift_m_s,
        1.0e-12,
    );
}

#[test]
fn fixed_plan_caches_dense_normal_diagonal() {
    let mask = Array2::from_elem((3, 3), true);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };

    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();

    assert_eq!(plan.cached_normal_diagonal_len(), plan.active_voxels());
    assert_eq!(plan.cached_sparse_lipschitz(), None);
}

#[test]
fn sparse_plan_caches_lipschitz_and_matches_direct_reconstruction() {
    let mask = Array2::from_elem((5, 5), true);
    let mut truth = Array2::zeros((5, 5));
    truth[[2, 2]] = 60.0;
    let samples = vec![
        horizontal_sample(0.0),
        horizontal_sample(-0.002),
        vertical_sample(0.0),
        vertical_sample(-0.002),
    ];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 80,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        sparsity_weight: 1.0e-5,
        sampling: ShiftSampling::Sparse {
            stride: 2,
            offset: 0,
        },
        prior: ShiftPrior::Sparse,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);
    let direct = reconstruct_sound_speed_shift(&measured, &mask, config).unwrap();
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();

    let planned = plan.reconstruct(&predicted).unwrap();

    assert_eq!(plan.cached_normal_diagonal_len(), plan.active_voxels());
    assert!(plan
        .cached_sparse_lipschitz()
        .is_some_and(|value| value.is_finite() && value > 0.0));
    assert_image_close(
        &planned.sound_speed_shift_m_s,
        &direct.sound_speed_shift_m_s,
        1.0e-12,
    );
}

#[test]
fn fixed_plan_rejects_invalid_frame_shift_vectors() {
    let mask = Array2::from_elem((3, 1), true);
    let samples = vec![horizontal_sample(0.0)];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let mut workspace = SoundSpeedShiftWorkspace::new();
    let mut plan_workspace = SoundSpeedShiftPlanWorkspace::new();
    let mut output = Array2::zeros((0, 0));

    let short = plan
        .reconstruct_with_workspace(&[], &mut workspace)
        .unwrap_err();
    let nonfinite = plan
        .reconstruct_with_workspace(&[f64::NAN], &mut workspace)
        .unwrap_err();
    let plan_workspace_short = plan
        .reconstruct_with_plan_workspace(&[], &mut plan_workspace)
        .unwrap_err();
    let in_place_short = plan
        .reconstruct_into_image(&[], &mut plan_workspace, &mut output)
        .unwrap_err();

    assert!(short.to_string().contains("expected 1 frame time shifts"));
    assert!(nonfinite.to_string().contains("nonfinite time shift"));
    assert!(plan_workspace_short
        .to_string()
        .contains("expected 1 frame time shifts"));
    assert!(in_place_short
        .to_string()
        .contains("expected 1 frame time shifts"));
}

#[test]
fn curved_array_plan_reuses_operator_across_repeated_frames() {
    let mask = Array2::from_elem((5, 5), true);
    let mut truth = Array2::zeros((5, 5));
    truth[[2, 2]] = 18.0;
    truth[[2, 3]] = 9.0;
    let array = CurvedArray2d {
        center_m: PlanarPoint { x_m: 0.0, y_m: 0.0 },
        radius_m: 0.004,
        first_angle_rad: 0.0,
        angular_pitch_rad: TAU / 8.0,
        element_count: 8,
    };
    let scan = CurvedArrayShiftScan::new(array, vec![3, 4]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 8,
        tikhonov_weight: 1.0e-8,
        propagation: ShiftPropagation::CircularArc {
            sagitta_m: 0.0005,
            segments: 8,
        },
        sensitivity: ShiftSensitivity::FiniteFrequency {
            wavelength_m: 0.001,
            support_radius_m: 0.002,
        },
        ..Default::default()
    };
    let plan = SoundSpeedShiftPlan::from_curved_array_scan(&scan, &mask, config).unwrap();
    let frame = plan.predict_time_shifts(&truth).unwrap();
    let scaled_frame = frame.iter().map(|value| 0.5 * *value).collect::<Vec<_>>();
    let mut workspace = SoundSpeedShiftWorkspace::new();

    let first = plan
        .reconstruct_with_workspace(&frame, &mut workspace)
        .unwrap();
    let retained_slots = workspace.allocated_slots();
    let retained_weights = plan.stored_weight_count();
    let second = plan
        .reconstruct_with_workspace(&scaled_frame, &mut workspace)
        .unwrap();

    assert_eq!(plan.rows_available(), scan.row_count());
    assert_eq!(plan.rows_used(), scan.row_count());
    assert_eq!(frame.len(), scan.row_count());
    assert_eq!(first.model_family, FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL);
    assert_eq!(
        second.model_family,
        FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL
    );
    assert_eq!(workspace.allocated_slots(), retained_slots);
    assert_eq!(plan.stored_weight_count(), retained_weights);
    assert!(
        first
            .sound_speed_shift_m_s
            .iter()
            .all(|value| value.is_finite())
            && second
                .sound_speed_shift_m_s
                .iter()
                .all(|value| value.is_finite())
    );
}
