use super::*;

#[test]
fn fixed_plan_workspace_reuses_sampled_rhs_and_solver_buffers() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(18.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 24,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);
    let direct = reconstruct_sound_speed_shift(&measured, &mask, config).unwrap();
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let mut workspace = SoundSpeedShiftPlanWorkspace::new();

    let first = plan
        .reconstruct_with_plan_workspace(&predicted, &mut workspace)
        .unwrap();
    let rhs_capacity = workspace.sampled_rhs_capacity();
    let retained_slots = workspace.allocated_slots();
    let retained_bytes = workspace.memory_bytes();
    let second = plan
        .reconstruct_with_plan_workspace(&predicted, &mut workspace)
        .unwrap();

    assert!(rhs_capacity >= plan.rows_used());
    assert_eq!(workspace.sampled_rhs_capacity(), rhs_capacity);
    assert_eq!(workspace.allocated_slots(), retained_slots);
    assert_eq!(workspace.memory_bytes(), retained_bytes);
    assert_image_close(
        &first.sound_speed_shift_m_s,
        &direct.sound_speed_shift_m_s,
        1.0e-12,
    );
    assert_eq!(first.sound_speed_shift_m_s, second.sound_speed_shift_m_s);
    workspace.clear();
    assert_eq!(workspace.sampled_rhs_capacity(), rhs_capacity);
}

#[test]
fn fixed_plan_reconstruct_into_image_reuses_output_allocation() {
    let mut mask = Array2::from_elem((3, 3), true);
    mask[[0, 0]] = false;
    let mut truth = Array2::zeros((3, 3));
    truth.fill(14.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 24,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let scaled = predicted
        .iter()
        .map(|value| 0.5 * *value)
        .collect::<Vec<_>>();
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let direct = plan.reconstruct(&predicted).unwrap();
    let scaled_direct = plan.reconstruct(&scaled).unwrap();
    let mut workspace = SoundSpeedShiftPlanWorkspace::new();
    let mut output = Array2::from_elem((1, 1), 999.0);

    {
        let first = plan
            .reconstruct_into_image(&predicted, &mut workspace, &mut output)
            .unwrap();

        assert_eq!(first.sound_speed_shift_m_s.dim(), mask.dim());
        assert!(!first.objective_history.is_empty());
        assert_eq!(first.rows_used, plan.rows_used());
        assert_eq!(first.rows_available, plan.rows_available());
        assert_eq!(first.active_voxels, plan.active_voxels());
        assert_eq!(first.model_family, SOUND_SPEED_SHIFT_MODEL);
        assert_eq!(first.sound_speed_shift_m_s[[0, 0]], 0.0);
        assert_image_close(
            first.sound_speed_shift_m_s,
            &direct.sound_speed_shift_m_s,
            1.0e-12,
        );
    }

    let retained_ptr = output.as_slice_memory_order().unwrap().as_ptr();
    output.fill(999.0);

    {
        let second = plan
            .reconstruct_into_image(&scaled, &mut workspace, &mut output)
            .unwrap();

        assert_eq!(
            second
                .sound_speed_shift_m_s
                .as_slice_memory_order()
                .unwrap()
                .as_ptr(),
            retained_ptr
        );
        assert_eq!(second.sound_speed_shift_m_s[[0, 0]], 0.0);
        assert_image_close(
            second.sound_speed_shift_m_s,
            &scaled_direct.sound_speed_shift_m_s,
            1.0e-12,
        );
    }
}
