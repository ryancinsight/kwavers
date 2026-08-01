//! Forward prediction sign contract.

use aequitas::systems::si::quantities::{Length, Time};
use leto::Array2;

use super::{
    SoundSpeedShiftConfig, SoundSpeedShiftField, SoundSpeedShiftSample,
    predict_sound_speed_time_shifts,
};
use kwavers_solver::inverse::same_aperture::PlanarPoint;

/// Linearized straight-ray sign contract:
/// `Δt = −(A·Δc) / c₀²`.
///
/// For a 3×1 active column, Δc = 20 m/s everywhere, and one horizontal ray
/// crossing all three cells (path integral = 3 × spacing_m = 0.003 m):
///
/// ```text
/// Δt = −(0.003 × 20) / c₀² ≈ −2.53 × 10⁻⁸ s
/// ```
///
/// The predicted shift must be negative (faster-than-reference path).
#[test]
fn forward_model_has_linearized_speed_shift_sign() {
    let mask = Array2::from_elem((3, 1), true);
    let mut shift = SoundSpeedShiftField::from_storage(Array2::zeros((3, 1)));
    shift.storage_mut().fill(20.0);
    let samples = vec![SoundSpeedShiftSample::new(
        PlanarPoint {
            x_m: -0.002,
            y_m: 0.0,
        },
        PlanarPoint {
            x_m: 0.002,
            y_m: 0.0,
        },
        Time::from_base(0.0),
    )];
    let config = SoundSpeedShiftConfig {
        spacing: Length::from_base(0.001),
        ..Default::default()
    };

    let predicted = predict_sound_speed_time_shifts(&shift, &samples, &mask, config).unwrap();
    let expected = -(3.0 * 0.001 * 20.0)
        / (config.reference_sound_speed.into_base() * config.reference_sound_speed.into_base());

    assert_eq!(predicted.len(), 1);
    let predicted_seconds = predicted[0].into_base();
    assert!(
        (predicted_seconds - expected).abs() <= 1.0e-15,
        "expected {expected:.12e}, got {:.12e}",
        predicted_seconds,
    );
    assert!(predicted_seconds < 0.0);
}
