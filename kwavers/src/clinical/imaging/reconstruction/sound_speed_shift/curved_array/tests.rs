use std::f64::consts::{FRAC_PI_2, PI, TAU};

use ndarray::Array2;

use super::{CurvedArray2d, CurvedArrayShiftScan};
use crate::clinical::imaging::reconstruction::sound_speed_shift::{
    predict_sound_speed_time_shifts, SoundSpeedShiftConfig,
};
use crate::solver::inverse::same_aperture::PlanarPoint;

#[test]
fn endpoint_arc_places_elements_on_declared_circle() {
    let array =
        CurvedArray2d::from_arc_endpoints(PlanarPoint { x_m: 0.0, y_m: 0.0 }, 0.02, PI, 0.0, 3);

    let elements = array.elements().unwrap();

    assert_eq!(elements.len(), 3);
    assert_close(elements[0].x_m, -0.02);
    assert_close(elements[0].y_m, 0.0);
    assert_close(elements[1].x_m, 0.0);
    assert_close(elements[1].y_m, 0.02);
    assert_close(elements[2].x_m, 0.02);
    assert_close(elements[2].y_m, 0.0);
    assert_close(array.aperture_angle_rad(), -PI);
}

#[test]
fn pitch_catch_rows_are_transmitter_major_and_offset_minor() {
    let array = CurvedArray2d {
        center_m: PlanarPoint { x_m: 0.0, y_m: 0.0 },
        radius_m: 1.0,
        first_angle_rad: 0.0,
        angular_pitch_rad: FRAC_PI_2,
        element_count: 4,
    };
    let scan = CurvedArrayShiftScan::new(array, vec![1, 2]);
    let shifts = (0..scan.row_count())
        .map(|idx| idx as f64 * 1.0e-9)
        .collect::<Vec<_>>();

    let elements = scan.elements().unwrap();
    let samples = scan.samples_with_time_shifts(&shifts).unwrap();

    assert_eq!(samples.len(), 8);
    assert_point(samples[0].transmitter, elements[0]);
    assert_point(samples[0].receiver, elements[1]);
    assert_eq!(samples[0].time_shift_s, 0.0);
    assert_point(samples[1].transmitter, elements[0]);
    assert_point(samples[1].receiver, elements[2]);
    assert_eq!(samples[1].time_shift_s, 1.0e-9);
    assert_point(samples[2].transmitter, elements[1]);
    assert_point(samples[2].receiver, elements[2]);
    assert_eq!(samples[2].time_shift_s, 2.0e-9);
}

#[test]
fn invalid_curved_scan_contract_is_rejected() {
    let array =
        CurvedArray2d::from_arc_endpoints(PlanarPoint { x_m: 0.0, y_m: 0.0 }, 0.01, PI, 0.0, 4);

    let zero_offset = CurvedArrayShiftScan::new(array, vec![0]);
    let duplicate_offset = CurvedArrayShiftScan::new(array, vec![1, 1]);
    let valid_scan = CurvedArrayShiftScan::new(array, vec![1]);

    assert!(zero_offset
        .samples()
        .unwrap_err()
        .to_string()
        .contains("receiver offset"));
    assert!(duplicate_offset
        .samples()
        .unwrap_err()
        .to_string()
        .contains("duplicated"));
    assert!(valid_scan
        .samples_with_time_shifts(&[])
        .unwrap_err()
        .to_string()
        .contains("expected 4 time shifts"));
}

#[test]
fn curved_array_samples_drive_straight_ray_shift_prediction() {
    let mask = Array2::from_elem((5, 5), true);
    let mut shift = Array2::zeros((5, 5));
    shift.fill(12.0);
    let array = CurvedArray2d {
        center_m: PlanarPoint { x_m: 0.0, y_m: 0.0 },
        radius_m: 0.004,
        first_angle_rad: 0.0,
        angular_pitch_rad: TAU / 8.0,
        element_count: 8,
    };
    let scan = CurvedArrayShiftScan::new(array, vec![4]);
    let samples = scan.samples().unwrap();
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        ..Default::default()
    };

    let predicted = predict_sound_speed_time_shifts(&shift, &samples, &mask, config).unwrap();

    assert_eq!(predicted.len(), scan.row_count());
    assert!(predicted.iter().all(|value| value.is_finite()));
    assert!(predicted.iter().any(|value| *value < 0.0));
    assert!(
        predicted.iter().map(|value| value.abs()).sum::<f64>() > 0.0,
        "curved-array straight-ray rows must intersect the active mask"
    );
}

fn assert_point(actual: PlanarPoint, expected: PlanarPoint) {
    assert_close(actual.x_m, expected.x_m);
    assert_close(actual.y_m, expected.y_m);
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "expected {expected:.12e}, got {actual:.12e}"
    );
}
