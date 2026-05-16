use super::*;

#[test]
fn batch_reconstruction_uses_compact_summaries_by_default() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(16.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 20,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let frame = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let scaled_frame = frame.iter().map(|value| 0.25 * *value).collect::<Vec<_>>();
    let frame_refs = [&frame[..], &scaled_frame[..]];
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let mut workspace = SoundSpeedShiftWorkspace::new();

    let batch = plan
        .reconstruct_frames_with_workspace(&frame_refs, &mut workspace)
        .unwrap();
    let direct = plan.reconstruct(&frame).unwrap();

    assert_eq!(batch.frames.len(), 2);
    assert_eq!(batch.rows_used, plan.rows_used());
    assert_eq!(batch.rows_available, plan.rows_available());
    assert_eq!(batch.active_voxels, plan.active_voxels());
    assert_eq!(batch.model_family, SOUND_SPEED_SHIFT_MODEL);
    assert!(batch.frames[0].objective_history.is_empty());
    assert_eq!(batch.frames[0].summary.frame_index, 0);
    assert!(batch.frames[0].summary.objective_iterations > 0);
    assert!(batch.frames[0].summary.objective_final <= batch.frames[0].summary.objective_initial);
    assert_image_close(
        batch.frames[0].sound_speed_shift_m_s.as_ref().unwrap(),
        &direct.sound_speed_shift_m_s,
        1.0e-12,
    );
}

#[test]
fn batch_reconstruction_retains_full_histories_when_requested() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(10.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 12,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let frame = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let frame_refs = [&frame[..]];
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let options = SoundSpeedShiftBatchConfig::default()
        .with_objective_history(SoundSpeedShiftObjectiveHistoryPolicy::Full);
    let mut workspace = SoundSpeedShiftWorkspace::new();

    let batch = plan
        .reconstruct_frames_with_options(&frame_refs, options, &mut workspace)
        .unwrap();

    assert_eq!(batch.frames.len(), 1);
    assert!(!batch.frames[0].objective_history.is_empty());
    assert_eq!(
        batch.frames[0].summary.objective_initial,
        *batch.frames[0].objective_history.first().unwrap()
    );
    assert_eq!(
        batch.frames[0].summary.objective_final,
        *batch.frames[0].objective_history.last().unwrap()
    );
    assert_eq!(
        batch.frames[0].summary.objective_iterations,
        batch.frames[0].objective_history.len() - 1
    );
    assert!(batch.frames[0].sound_speed_shift_m_s.is_some());
}

#[test]
fn batch_reconstruction_rejects_invalid_frame_batches() {
    let mask = Array2::from_elem((3, 1), true);
    let samples = vec![horizontal_sample(0.0)];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let mut workspace = SoundSpeedShiftWorkspace::new();
    let empty: [&[f64]; 0] = [];
    let short = [&[][..]];
    let nonfinite = [&[f64::NAN][..]];

    let empty_err = plan
        .reconstruct_frames_with_workspace(&empty, &mut workspace)
        .unwrap_err();
    let short_err = plan
        .reconstruct_frames_with_workspace(&short, &mut workspace)
        .unwrap_err();
    let nonfinite_err = plan
        .reconstruct_frames_with_workspace(&nonfinite, &mut workspace)
        .unwrap_err();

    assert!(empty_err
        .to_string()
        .contains("requires at least one frame"));
    assert!(short_err.to_string().contains("batch frame 0 expected 1"));
    assert!(nonfinite_err
        .to_string()
        .contains("batch frame 0 row 0 has nonfinite time shift"));
}
