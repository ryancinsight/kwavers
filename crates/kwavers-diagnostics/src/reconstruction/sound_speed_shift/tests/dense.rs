//! Dense Tikhonov/H1 PCG reconstruction tests.

use leto::Array2;

use super::{
    attach_time_shifts, horizontal_samples, predict_sound_speed_time_shifts,
    reconstruct_sound_speed_shift, reconstruct_sound_speed_shift_with_workspace, ShiftPrior,
    ShiftSampling, SoundSpeedShiftConfig, SoundSpeedShiftWorkspace, SOUND_SPEED_SHIFT_MODEL,
};

/// Dense PCG recovers a uniform 20 m/s shift on a fully-observed 3×3 grid.
///
/// Three horizontal rays, each crossing a full 3-cell row, give a
/// rank-3 system; with zero Tikhonov and zero smoothness weight, PCG
/// converges to the exact 20 m/s solution in ≤ 32 iterations.
#[test]
fn dense_prior_recovers_uniform_sound_speed_shift() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(20.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let mut config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 24,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        sampling: ShiftSampling::Dense,
        prior: ShiftPrior::Dense,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);
    config.iterations = 32;

    let image = reconstruct_sound_speed_shift(&measured, &mask, config).unwrap();

    assert_eq!(image.model_family, SOUND_SPEED_SHIFT_MODEL);
    assert_eq!(image.rows_used, 3);
    assert_eq!(image.active_voxels, 9);
    for value in image.sound_speed_shift_m_s.iter() {
        assert!(
            (*value - 20.0).abs() <= 1.0e-4,
            "dense reconstruction value {value} differs from 20 m/s"
        );
    }
    assert!(
        image.objective_history.last().unwrap() <= image.objective_history.first().unwrap(),
        "objective did not decrease"
    );
}

/// Workspace slot count and byte footprint are preserved across two successive
/// Dense calls with identical geometry.  Solution vectors must be equal.
#[test]
fn reconstruction_workspace_reuses_solver_buffers() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(20.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 24,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        sampling: ShiftSampling::Dense,
        prior: ShiftPrior::Dense,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);
    let mut workspace = SoundSpeedShiftWorkspace::new();

    let first =
        reconstruct_sound_speed_shift_with_workspace(&measured, &mask, config, &mut workspace)
            .unwrap();
    let retained_slots = workspace.allocated_slots();
    let retained_bytes = workspace.memory_bytes();
    let second =
        reconstruct_sound_speed_shift_with_workspace(&measured, &mask, config, &mut workspace)
            .unwrap();

    assert!(retained_slots > 0);
    assert_eq!(workspace.allocated_slots(), retained_slots);
    assert_eq!(workspace.memory_bytes(), retained_bytes);
    assert_eq!(first.sound_speed_shift_m_s, second.sound_speed_shift_m_s);
    workspace.clear();
    assert_eq!(workspace.allocated_slots(), retained_slots);
}
