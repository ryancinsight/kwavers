//! Tests for plane wave delay calculation.

use super::config::PlaneWaveConfig;
use super::processor::PlaneWave;
use approx::assert_relative_eq;
use ndarray::Array1;
use std::f64::consts::PI;

#[test]
fn test_plane_wave_transmission_delays() {
    let positions = vec![-0.002, -0.001, 0.0, 0.001, 0.002];
    let config = PlaneWaveConfig {
        element_positions: positions,
        sound_speed: 1540.0,
        ..Default::default()
    };
    let pw = PlaneWave::new(config);

    let delays_0deg = pw.transmission_delays(0.0).unwrap();
    assert_eq!(delays_0deg.len(), 5);
    for &delay in delays_0deg.iter() {
        assert_relative_eq!(delay, 0.0, epsilon = 1e-12);
    }

    let theta = 5.0 * PI / 180.0;
    let delays_5deg = pw.transmission_delays(theta).unwrap();
    assert_eq!(delays_5deg.len(), 5);

    let expected_delay_per_mm = -theta.sin() / 1540.0;
    assert_relative_eq!(
        delays_5deg[1] - delays_5deg[0],
        expected_delay_per_mm * 0.001,
        epsilon = 1e-9
    );
}

#[test]
fn test_beamforming_delays() {
    let positions = vec![-0.001, 0.0, 0.001];
    let config = PlaneWaveConfig {
        element_positions: positions,
        sound_speed: 1540.0,
        ..Default::default()
    };
    let pw = PlaneWave::new(config);

    let delays = pw.beamforming_delays(0.0, 0.02, 0.0).unwrap();
    assert_eq!(delays.len(), 3);

    let expected = 0.02 / 1540.0;
    for &delay in delays.iter() {
        assert_relative_eq!(delay, expected, epsilon = 1e-9);
    }
}

#[test]
fn test_apodization_weights() {
    let positions: Vec<f64> = (0..128).map(|i| (i as f64 - 63.5) * 0.00011).collect();
    let config = PlaneWaveConfig {
        element_positions: positions,
        f_number: Some(1.5),
        ..Default::default()
    };
    let pw = PlaneWave::new(config);

    let weights = pw.apodization_weights(0.0, 0.02).unwrap();
    assert_eq!(weights.len(), 128);

    let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_relative_eq!(max_weight, 1.0, epsilon = 2e-4);

    assert_relative_eq!(weights[0], weights[127], epsilon = 1e-6);
    assert_relative_eq!(weights[10], weights[117], epsilon = 1e-6);
}

#[test]
fn test_functional_ultrasound_config() {
    let positions: Vec<f64> = (0..128).map(|i| (i as f64 - 63.5) * 0.00011).collect();
    let pw = PlaneWave::functional_ultrasound(positions);

    assert_eq!(pw.num_angles(), 11);

    let angles_deg = pw.angles_degrees();
    assert_relative_eq!(angles_deg[0], -10.0, epsilon = 0.1);
    assert_relative_eq!(angles_deg[10], 10.0, epsilon = 0.1);

    let frame_rate = pw.compounded_frame_rate(5500.0);
    assert_relative_eq!(frame_rate, 500.0, epsilon = 0.1);
}

#[test]
fn test_delay_surface() {
    let positions = vec![-0.001, 0.0, 0.001];
    let config = PlaneWaveConfig {
        element_positions: positions,
        sound_speed: 1540.0,
        ..Default::default()
    };
    let pw = PlaneWave::new(config);

    let x_pixels = Array1::from_vec(vec![-0.005, 0.0, 0.005]);
    let y_pixels = Array1::from_vec(vec![0.01, 0.02]);

    let surface = pw.delay_surface(&x_pixels, &y_pixels, 0.0).unwrap();
    assert_eq!(surface.dim(), (3, 6));

    let y = 0.01;
    let expected_depth_delay = y / 1540.0;
    assert_relative_eq!(surface[[1, 1]], expected_depth_delay, epsilon = 1e-9);
}
