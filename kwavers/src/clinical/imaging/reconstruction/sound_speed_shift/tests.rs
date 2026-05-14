use ndarray::Array2;

use super::{
    operator::SoundSpeedShiftOperator, predict_sound_speed_time_shifts,
    reconstruct_sound_speed_shift, reconstruct_sound_speed_shift_with_workspace, ShiftPrior,
    ShiftPropagation, ShiftSampling, ShiftSensitivity, SoundSpeedShiftConfig,
    SoundSpeedShiftSample, SoundSpeedShiftWorkspace, SOUND_SPEED_SHIFT_MODEL,
};
use crate::solver::inverse::same_aperture::PlanarPoint;

#[test]
fn forward_model_has_linearized_speed_shift_sign() {
    let mask = Array2::from_elem((3, 1), true);
    let mut shift = Array2::zeros((3, 1));
    shift.fill(20.0);
    let samples = vec![SoundSpeedShiftSample::new(
        PlanarPoint {
            x_m: -0.002,
            y_m: 0.0,
        },
        PlanarPoint {
            x_m: 0.002,
            y_m: 0.0,
        },
        0.0,
    )];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };

    let predicted = predict_sound_speed_time_shifts(&shift, &samples, &mask, config).unwrap();
    let expected = -(3.0 * 0.001 * 20.0)
        / (config.reference_sound_speed_m_s * config.reference_sound_speed_m_s);

    assert_eq!(predicted.len(), 1);
    assert!(
        (predicted[0] - expected).abs() <= 1.0e-15,
        "expected {expected:.12e}, got {:.12e}",
        predicted[0],
    );
    assert!(predicted[0] < 0.0);
}

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
    for value in &image.sound_speed_shift_m_s {
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

#[test]
fn sparse_sampling_and_prior_localize_crossing_shift() {
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
        iterations: 160,
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

    let image = reconstruct_sound_speed_shift(&measured, &mask, config).unwrap();
    let center = image.sound_speed_shift_m_s[[2, 2]];
    let neighbor = image.sound_speed_shift_m_s[[2, 1]].max(image.sound_speed_shift_m_s[[1, 2]]);

    assert_eq!(image.rows_available, 4);
    assert_eq!(image.rows_used, 2);
    assert!(center > 0.0, "center perturbation was not recovered");
    assert!(
        center > neighbor,
        "sparse crossing row should give center dominance, center={center}, neighbor={neighbor}"
    );
}

#[test]
fn invalid_sparse_sampling_is_rejected() {
    let mask = Array2::from_elem((1, 1), true);
    let samples = vec![horizontal_sample(0.0)];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        sampling: ShiftSampling::Sparse {
            stride: 0,
            offset: 0,
        },
        ..Default::default()
    };

    let err = reconstruct_sound_speed_shift(&samples, &mask, config).unwrap_err();
    assert!(err.to_string().contains("Sparse sampling requires stride"));
}

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

#[test]
fn operator_stores_only_crossed_ray_segments() {
    let mask = Array2::from_elem((32, 32), true);
    let samples = vec![SoundSpeedShiftSample::new(
        PlanarPoint {
            x_m: -0.024,
            y_m: -0.024,
        },
        PlanarPoint {
            x_m: 0.024,
            y_m: 0.024,
        },
        0.0,
    )];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };

    let operator = SoundSpeedShiftOperator::new(&samples, &mask, config).unwrap();

    assert_eq!(operator.rows(), 1);
    assert_eq!(operator.cols(), 32 * 32);
    assert!(
        operator.stored_segment_count() <= 64,
        "diagonal row should store O(nx + ny) crossed cells, stored {}",
        operator.stored_segment_count()
    );
    assert!(
        operator.stored_segment_count() < operator.cols() / 8,
        "ray storage should not materialize the full active mask"
    );
}

#[test]
fn curved_ray_prediction_has_longer_uniform_path_than_straight_chord() {
    let mask = Array2::from_elem((9, 9), true);
    let mut shift = Array2::zeros((9, 9));
    shift.fill(10.0);
    let samples = vec![SoundSpeedShiftSample::new(
        PlanarPoint {
            x_m: -0.002,
            y_m: 0.0,
        },
        PlanarPoint {
            x_m: 0.002,
            y_m: 0.0,
        },
        0.0,
    )];
    let straight = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };
    let curved = SoundSpeedShiftConfig {
        propagation: ShiftPropagation::CircularArc {
            sagitta_m: 0.001,
            segments: 16,
        },
        ..straight
    };

    let straight_shift =
        predict_sound_speed_time_shifts(&shift, &samples, &mask, straight).unwrap();
    let curved_shift = predict_sound_speed_time_shifts(&shift, &samples, &mask, curved).unwrap();

    assert!(straight_shift[0] < 0.0);
    assert!(curved_shift[0] < straight_shift[0]);
}

#[test]
fn finite_frequency_sensitivity_detects_off_axis_shift() {
    let mask = Array2::from_elem((7, 7), true);
    let mut shift = Array2::zeros((7, 7));
    shift[[3, 4]] = 80.0;
    let samples = vec![SoundSpeedShiftSample::new(
        PlanarPoint {
            x_m: -0.002,
            y_m: 0.0,
        },
        PlanarPoint {
            x_m: 0.002,
            y_m: 0.0,
        },
        0.0,
    )];
    let geometric = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };
    let finite = SoundSpeedShiftConfig {
        sensitivity: ShiftSensitivity::FiniteFrequency {
            wavelength_m: 0.001,
            support_radius_m: 0.002,
        },
        ..geometric
    };

    let geometric_shift =
        predict_sound_speed_time_shifts(&shift, &samples, &mask, geometric).unwrap();
    let finite_shift = predict_sound_speed_time_shifts(&shift, &samples, &mask, finite).unwrap();

    assert_eq!(geometric_shift[0], 0.0);
    assert!(
        finite_shift[0] < 0.0,
        "finite-frequency tube should assign sensitivity to the off-axis cell"
    );
}

#[test]
fn invalid_curved_ray_and_finite_frequency_config_are_rejected() {
    let mask = Array2::from_elem((3, 3), true);
    let samples = vec![horizontal_sample(0.0)];
    let invalid_arc = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        propagation: ShiftPropagation::CircularArc {
            sagitta_m: 0.0,
            segments: 1,
        },
        ..Default::default()
    };
    let invalid_frequency = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        sensitivity: ShiftSensitivity::FiniteFrequency {
            wavelength_m: 0.0,
            support_radius_m: 0.002,
        },
        ..Default::default()
    };

    assert!(reconstruct_sound_speed_shift(&samples, &mask, invalid_arc)
        .unwrap_err()
        .to_string()
        .contains("Circular-arc propagation"));
    assert!(
        reconstruct_sound_speed_shift(&samples, &mask, invalid_frequency)
            .unwrap_err()
            .to_string()
            .contains("Finite-frequency sensitivity")
    );
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

fn vertical_sample(x_m: f64) -> SoundSpeedShiftSample {
    SoundSpeedShiftSample::new(
        PlanarPoint { x_m, y_m: -0.004 },
        PlanarPoint { x_m, y_m: 0.004 },
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
