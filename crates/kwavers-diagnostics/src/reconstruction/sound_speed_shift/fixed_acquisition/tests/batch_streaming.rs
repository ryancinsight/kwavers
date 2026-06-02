use super::*;
use kwavers_core::error::KwaversError;

#[test]
fn batch_streaming_reuses_image_view_without_retaining_frames() {
    let mut mask = Array2::from_elem((3, 3), true);
    mask[[0, 0]] = false;
    let mut truth = Array2::zeros((3, 3));
    truth.fill(9.0);
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
    let expected = vec![
        plan.reconstruct(&frame).unwrap().sound_speed_shift_m_s,
        plan.reconstruct(&scaled_frame)
            .unwrap()
            .sound_speed_shift_m_s,
    ];
    let rows_used = plan.rows_used();
    let rows_available = plan.rows_available();
    let active_voxels = plan.active_voxels();
    let mut workspace = SoundSpeedShiftPlanWorkspace::new();
    let mut summaries = Vec::new();
    let mut image_ptrs = Vec::new();
    let mut objective_lengths = Vec::new();

    let stream = plan
        .reconstruct_frames_streaming_with_plan_workspace(
            &frame_refs,
            &mut workspace,
            |summary, view| {
                let frame_index = summary.frame_index;
                assert_eq!(view.rows_used, rows_used);
                assert_eq!(view.rows_available, rows_available);
                assert_eq!(view.active_voxels, active_voxels);
                assert_eq!(view.model_family, SOUND_SPEED_SHIFT_MODEL);
                assert_eq!(view.sound_speed_shift_m_s[[0, 0]], 0.0);
                assert!(!view.objective_history.is_empty());
                assert_image_close(view.sound_speed_shift_m_s, &expected[frame_index], 1.0e-12);
                image_ptrs.push(
                    view.sound_speed_shift_m_s
                        .as_slice_memory_order()
                        .unwrap()
                        .as_ptr() as usize,
                );
                objective_lengths.push(view.objective_history.len());
                summaries.push(summary);
                Ok(())
            },
        )
        .unwrap();

    assert_eq!(stream.frames_processed, 2);
    assert_eq!(stream.rows_used, rows_used);
    assert_eq!(stream.rows_available, rows_available);
    assert_eq!(stream.active_voxels, active_voxels);
    assert_eq!(stream.model_family, SOUND_SPEED_SHIFT_MODEL);
    assert_eq!(summaries.len(), 2);
    assert_eq!(summaries[0].frame_index, 0);
    assert_eq!(summaries[1].frame_index, 1);
    assert_eq!(image_ptrs.len(), 2);
    assert_eq!(image_ptrs[0], image_ptrs[1]);
    assert!(objective_lengths.iter().all(|length| *length > 0));

    let rhs_capacity = workspace.sampled_rhs_capacity();
    let retained_slots = workspace.allocated_slots();
    let repeated = plan
        .reconstruct_frames_streaming_with_plan_workspace(&frame_refs, &mut workspace, |_, view| {
            assert_eq!(view.rows_used, rows_used);
            Ok(())
        })
        .unwrap();

    assert_eq!(repeated.frames_processed, 2);
    assert_eq!(workspace.sampled_rhs_capacity(), rhs_capacity);
    assert_eq!(workspace.allocated_slots(), retained_slots);
}

#[test]
fn batch_streaming_propagates_callback_error_and_stops() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(7.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 8,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let frame = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let scaled_frame = frame.iter().map(|value| 0.25 * *value).collect::<Vec<_>>();
    let frame_refs = [&frame[..], &scaled_frame[..]];
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let mut seen = 0usize;

    let error = plan
        .reconstruct_frames_streaming(&frame_refs, |summary, view| {
            seen += 1;
            assert_eq!(summary.frame_index, 0);
            assert!(!view.objective_history.is_empty());
            Err(KwaversError::InvalidInput(
                "stream callback stop".to_owned(),
            ))
        })
        .unwrap_err();

    assert_eq!(seen, 1);
    assert!(error.to_string().contains("stream callback stop"));
}
