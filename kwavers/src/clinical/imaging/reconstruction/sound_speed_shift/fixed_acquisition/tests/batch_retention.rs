use super::*;

#[test]
fn batch_discard_image_policy_retains_summaries_without_images() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(12.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 16,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let frame = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let scaled_frame = frame.iter().map(|value| 0.5 * *value).collect::<Vec<_>>();
    let frame_refs = [&frame[..], &scaled_frame[..]];
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let options = SoundSpeedShiftBatchConfig::default()
        .with_image_retention(SoundSpeedShiftBatchImageRetention::Discard)
        .with_objective_history(SoundSpeedShiftObjectiveHistoryPolicy::Full);
    let mut workspace = SoundSpeedShiftPlanWorkspace::new();

    let batch = plan
        .reconstruct_frames_with_plan_workspace_and_options(&frame_refs, options, &mut workspace)
        .unwrap();
    let rhs_capacity = workspace.sampled_rhs_capacity();
    let retained_slots = workspace.allocated_slots();
    let repeated = plan
        .reconstruct_frames_with_plan_workspace_and_options(&frame_refs, options, &mut workspace)
        .unwrap();

    assert_eq!(batch.frames.len(), 2);
    assert_eq!(batch.rows_used, plan.rows_used());
    assert_eq!(batch.rows_available, plan.rows_available());
    assert_eq!(batch.active_voxels, plan.active_voxels());
    assert_eq!(batch.model_family, SOUND_SPEED_SHIFT_MODEL);
    assert!(batch
        .frames
        .iter()
        .all(|frame| frame.sound_speed_shift_m_s.is_none()));
    assert!(batch
        .frames
        .iter()
        .all(|frame| !frame.objective_history.is_empty()));
    assert_eq!(batch.frames[0].summary.frame_index, 0);
    assert_eq!(batch.frames[1].summary.frame_index, 1);
    assert!(batch.frames[0].summary.objective_final <= batch.frames[0].summary.objective_initial);
    assert!(batch.frames[1].summary.objective_final <= batch.frames[1].summary.objective_initial);
    assert_eq!(workspace.sampled_rhs_capacity(), rhs_capacity);
    assert_eq!(workspace.allocated_slots(), retained_slots);
    assert_eq!(repeated.frames[0].summary, batch.frames[0].summary);
    assert!(repeated.frames[0].sound_speed_shift_m_s.is_none());
}

#[test]
fn batch_retain_policy_writes_images_through_plan_workspace() {
    let mut mask = Array2::from_elem((3, 3), true);
    mask[[0, 0]] = false;
    let mut truth = Array2::zeros((3, 3));
    truth.fill(11.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 16,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let frame = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let frame_refs = [&frame[..]];
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let direct = plan.reconstruct(&frame).unwrap();
    let mut workspace = SoundSpeedShiftPlanWorkspace::new();

    let batch = plan
        .reconstruct_frames_with_plan_workspace(&frame_refs, &mut workspace)
        .unwrap();

    let retained = batch.frames[0].sound_speed_shift_m_s.as_ref().unwrap();
    assert_eq!(retained[[0, 0]], 0.0);
    assert_image_close(retained, &direct.sound_speed_shift_m_s, 1.0e-12);
    assert!(workspace.sampled_rhs_capacity() >= plan.rows_used());
}
