//! Tests for plane wave delay calculation.

#![cfg_attr(test, expect(clippy::unwrap_used, reason = "ratchet KWAVERS-UNWRAP-1"))]

use super::config::UltrafastPlaneWaveConfig;
use super::processor::UltrafastPlaneWave;
use aequitas::systems::si::quantities::{Angle, Dimensionless, Length, Velocity};
use aequitas::systems::si::units::{Meter, MeterPerSecond, Radian};
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use leto::Array1;
use std::f64::consts::PI;

fn meters(value: f64) -> Length<f64> {
    Length::from_unit::<Meter>(value)
}

fn radians(value: f64) -> Angle<f64> {
    Angle::from_unit::<Radian>(value)
}

#[test]
fn test_plane_wave_transmission_delays() {
    let positions = vec![-0.002, -0.001, 0.0, 0.001, 0.002];
    let config = UltrafastPlaneWaveConfig {
        element_positions: positions.into_iter().map(meters).collect(),
        sound_speed: Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        ..Default::default()
    };
    let pw = UltrafastPlaneWave::new(config);

    let delays_0deg = pw.transmission_delays(radians(0.0)).unwrap();
    assert_eq!(delays_0deg.len(), 5);
    for &delay in delays_0deg.iter() {
        assert_relative_eq!(delay, 0.0, epsilon = 1e-12);
    }

    let theta = 5.0 * PI / 180.0;
    let delays_5deg = pw.transmission_delays(radians(theta)).unwrap();
    assert_eq!(delays_5deg.len(), 5);

    let expected_delay_per_mm = -theta.sin() / SOUND_SPEED_TISSUE;
    assert_relative_eq!(
        delays_5deg[1] - delays_5deg[0],
        expected_delay_per_mm * 0.001,
        epsilon = 1e-9
    );
}

#[test]
fn test_beamforming_delays() {
    let positions = vec![-0.001, 0.0, 0.001];
    let config = UltrafastPlaneWaveConfig {
        element_positions: positions.into_iter().map(meters).collect(),
        sound_speed: Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        ..Default::default()
    };
    let pw = UltrafastPlaneWave::new(config);

    let delays = pw
        .beamforming_delays(meters(0.0), meters(0.02), radians(0.0))
        .unwrap();
    assert_eq!(delays.len(), 3);

    let expected = 0.02 / SOUND_SPEED_TISSUE;
    for &delay in delays.iter() {
        assert_relative_eq!(delay, expected, epsilon = 1e-9);
    }
}

#[test]
fn test_apodization_weights() {
    let positions: Vec<Length<f64>> = (0..128)
        .map(|i| meters((i as f64 - 63.5) * 0.00011))
        .collect();
    let config = UltrafastPlaneWaveConfig {
        element_positions: positions,
        f_number: Some(Dimensionless::from_base(1.5)),
        ..Default::default()
    };
    let pw = UltrafastPlaneWave::new(config);

    let weights = pw.apodization_weights(meters(0.0), meters(0.02)).unwrap();
    assert_eq!(weights.len(), 128);

    let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_relative_eq!(max_weight, 1.0, epsilon = 2e-4);

    assert_relative_eq!(weights[0], weights[127], epsilon = 1e-6);
    assert_relative_eq!(weights[10], weights[117], epsilon = 1e-6);
}

#[test]
fn test_functional_ultrasound_config() {
    let positions: Vec<Length<f64>> = (0..128)
        .map(|i| meters((i as f64 - 63.5) * 0.00011))
        .collect();
    let pw = UltrafastPlaneWave::functional_ultrasound(positions);

    assert_eq!(pw.num_angles(), 11);

    let angles = pw.angles();
    assert_relative_eq!(
        angles[0].in_unit::<Radian>(),
        -10.0_f64.to_radians(),
        epsilon = 1e-12
    );
    assert_relative_eq!(
        angles[10].in_unit::<Radian>(),
        10.0_f64.to_radians(),
        epsilon = 1e-12
    );

    let frame_rate = pw.compounded_frame_rate(
        aequitas::systems::si::quantities::Frequency::from_base(5500.0),
    );
    assert_relative_eq!(frame_rate.into_base(), 500.0, epsilon = 0.1);
}

#[test]
fn test_delay_surface() {
    let positions = vec![-0.001, 0.0, 0.001];
    let config = UltrafastPlaneWaveConfig {
        element_positions: positions.into_iter().map(meters).collect(),
        sound_speed: Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        ..Default::default()
    };
    let pw = UltrafastPlaneWave::new(config);

    let x_pixels = Array1::from_vec(3, vec![meters(-0.005), meters(0.0), meters(0.005)]).unwrap();
    let y_pixels = Array1::from_vec(2, vec![meters(0.01), meters(0.02)]).unwrap();

    let surface = pw
        .delay_surface(&x_pixels, &y_pixels, radians(0.0))
        .unwrap();
    assert_eq!(surface.shape(), [3, 6]);

    let y = 0.01;
    let expected_depth_delay = y / SOUND_SPEED_TISSUE;
    assert_relative_eq!(surface[[1, 1]], expected_depth_delay, epsilon = 1e-9);
}
