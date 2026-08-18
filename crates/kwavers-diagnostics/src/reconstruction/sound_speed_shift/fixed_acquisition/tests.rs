use std::f64::consts::TAU;

use leto::Array2;

use super::SoundSpeedShiftPlan;
use crate::reconstruction::sound_speed_shift::{
    predict_sound_speed_time_shifts, reconstruct_sound_speed_shift, CurvedArray2d,
    CurvedArrayShiftScan, ShiftPropagation, ShiftSensitivity, SoundSpeedShiftBatchConfig,
    SoundSpeedShiftConfig, SoundSpeedShiftObjectiveHistoryPolicy, SoundSpeedShiftSample,
    SoundSpeedShiftWorkspace, FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL, SOUND_SPEED_SHIFT_MODEL,
};
use kwavers_solver::inverse::same_aperture::PlanarPoint;

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
fn fixed_plan_rejects_invalid_frame_shift_vectors() {
    let mask = Array2::from_elem((3, 1), true);
    let samples = vec![horizontal_sample(0.0)];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };
    let plan = SoundSpeedShiftPlan::new(samples, &mask, config).unwrap();
    let mut workspace = SoundSpeedShiftWorkspace::new();

    let short = plan
        .reconstruct_with_workspace(&[], &mut workspace)
        .unwrap_err();
    let nonfinite = plan
        .reconstruct_with_workspace(&[f64::NAN], &mut workspace)
        .unwrap_err();

    assert!(short.to_string().contains("expected 1 frame time shifts"));
    assert!(nonfinite.to_string().contains("nonfinite time shift"));
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
        batch.frames[0]
            .sound_speed_shift_m_s
            .as_ref()
            .expect("default batch retention must retain reconstructed image"),
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

fn horizontal_samples(y_values: &[f64]) -> Vec<SoundSpeedShiftSample> {
    y_values.iter().map(|y| horizontal_sample(*y)).collect()
}

fn horizontal_sample(y_m: f64) -> SoundSpeedShiftSample {
    SoundSpeedShiftSample::new(
        PlanarPoint { x_m: -0.004, y_m },
        PlanarPoint { x_m: 0.004, y_m },
        0.0,
    )
}

fn attach_time_shifts(
    samples: &[SoundSpeedShiftSample],
    time_shifts: &[f64],
) -> Vec<SoundSpeedShiftSample> {
    samples
        .iter()
        .zip(time_shifts.iter())
        .map(|(sample, time_shift_s)| SoundSpeedShiftSample {
            time_shift_s: *time_shift_s,
            ..*sample
        })
        .collect()
}

fn assert_image_close(actual: &Array2<f64>, expected: &Array2<f64>, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for ((idx, actual), expected) in actual.indexed_iter().zip(expected.iter()) {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "image value at {idx:?} differs: expected {expected:.12e}, got {actual:.12e}"
        );
    }
}
