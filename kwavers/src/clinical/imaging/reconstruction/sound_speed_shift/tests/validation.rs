//! Operator geometry, sensitivity model, and invalid configuration rejection tests.

use ndarray::Array2;

use super::{
    horizontal_sample, predict_sound_speed_time_shifts, reconstruct_sound_speed_shift,
    ShiftPropagation, ShiftSensitivity, SoundSpeedShiftConfig, SoundSpeedShiftSample,
};
use crate::{
    clinical::imaging::reconstruction::sound_speed_shift::operator::SoundSpeedShiftOperator,
    solver::inverse::same_aperture::PlanarPoint,
};

/// Diagonal straight-ray should store O(nx + ny) crossed cells, not the full
/// active mask.
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

/// Circular-arc path through a uniform shift field has a longer effective path
/// integral than the straight chord, so the predicted time shift is larger in
/// magnitude.
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

/// Finite-frequency sensitivity assigns weight to an off-axis cell that
/// geometric ray sensitivity misses entirely.
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

/// Invalid curved-ray (zero sagitta, segments < 2) and invalid finite-frequency
/// (zero wavelength) configs are each rejected with a descriptive error.
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
