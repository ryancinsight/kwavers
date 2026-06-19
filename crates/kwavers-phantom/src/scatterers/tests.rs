//! Value-semantic tests for point-scatterer RF synthesis. Expected delays and
//! amplitudes are derived analytically from the monostatic echo model.

use super::{RfSynthesisConfig, ScattererCloud};

const C: f64 = 1540.0; // m/s
const FS: f64 = 40e6; // 40 MHz

fn cfg(num_samples: usize) -> RfSynthesisConfig {
    RfSynthesisConfig {
        sound_speed: C,
        sampling_frequency: FS,
        num_samples,
        min_distance: 0.0,
    }
}

/// Single scatterer, single element: echo lands at sample round(2r/c·fs) with
/// amplitude a/r² (impulse pulse isolates the value).
#[test]
fn single_scatterer_lands_at_round_trip_delay_with_inverse_square_amplitude() {
    let d = 0.02; // 20 mm depth on axis
    let amp = 3.0;
    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, d]], &[amp]).unwrap();
    let element = [[0.0, 0.0, 0.0]];
    let pulse = [1.0]; // unit impulse

    let n = 4096;
    let rf = cloud.synthesize_rf(&element, &pulse, &cfg(n)).unwrap();

    let n_delay = ((2.0 * d / C) * FS).round() as usize;
    let expected_amp = amp / (d * d);
    assert!(
        (rf[[0, n_delay]] - expected_amp).abs() < 1e-9,
        "expected {expected_amp} at sample {n_delay}, got {}",
        rf[[0, n_delay]]
    );
    // Energy is concentrated at the delay sample only (impulse pulse).
    let total: f64 = rf.row(0).iter().map(|&v| v.abs()).sum();
    assert!((total - expected_amp).abs() < 1e-9, "echo must be a single nonzero sample");
}

#[test]
fn echoes_superpose_linearly() {
    let cloud = ScattererCloud::from_points(
        &[[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]],
        &[2.0, 5.0],
    )
    .unwrap();
    let element = [[0.0, 0.0, 0.0]];
    let pulse = [1.0];
    let n = 4096;
    let rf = cloud.synthesize_rf(&element, &pulse, &cfg(n)).unwrap();

    for (d, a) in [(0.01, 2.0), (0.02, 5.0)] {
        let n_delay = ((2.0 * d / C) * FS).round() as usize;
        assert!((rf[[0, n_delay]] - a / (d * d)).abs() < 1e-9, "scatterer at {d} m");
    }
}

#[test]
fn amplitude_scales_linearly_with_reflectivity() {
    let element = [[0.0, 0.0, 0.0]];
    let pulse = [1.0];
    let n = 4096;
    let a = ScattererCloud::from_points(&[[0.0, 0.0, 0.015]], &[1.0])
        .unwrap()
        .synthesize_rf(&element, &pulse, &cfg(n))
        .unwrap();
    let b = ScattererCloud::from_points(&[[0.0, 0.0, 0.015]], &[2.0])
        .unwrap()
        .synthesize_rf(&element, &pulse, &cfg(n))
        .unwrap();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((2.0 * x - y).abs() < 1e-12, "RF must be linear in amplitude");
    }
}

#[test]
fn zero_amplitude_scatterer_yields_zero_rf() {
    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, 0.02]], &[0.0]).unwrap();
    let rf = cloud
        .synthesize_rf(&[[0.0, 0.0, 0.0]], &[1.0], &cfg(2048))
        .unwrap();
    assert!(rf.iter().all(|&v| v == 0.0));
}

#[test]
fn pulse_is_placed_starting_at_the_delay_sample() {
    // Multi-sample pulse: RF should be the pulse shifted by n_delay, scaled by a/r².
    let d = 0.01;
    let amp = 1.0;
    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, d]], &[amp]).unwrap();
    let pulse = [0.5, -1.0, 0.25];
    let n = 4096;
    let rf = cloud
        .synthesize_rf(&[[0.0, 0.0, 0.0]], &pulse, &cfg(n))
        .unwrap();
    let n_delay = ((2.0 * d / C) * FS).round() as usize;
    let scale = amp / (d * d);
    for (k, &p) in pulse.iter().enumerate() {
        assert!((rf[[0, n_delay + k]] - scale * p).abs() < 1e-9, "pulse sample {k}");
    }
}

#[test]
fn min_distance_guard_skips_near_scatterers() {
    // Scatterer essentially at the element: skipped by min_distance, no NaN/inf.
    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, 1e-9]], &[1.0]).unwrap();
    let mut config = cfg(1024);
    config.min_distance = 1e-3;
    let rf = cloud
        .synthesize_rf(&[[0.0, 0.0, 0.0]], &[1.0], &config)
        .unwrap();
    assert!(rf.iter().all(|&v| v == 0.0 && v.is_finite()));
}

#[test]
fn rejects_invalid_config_and_mismatched_lengths() {
    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, 0.02]], &[1.0]).unwrap();
    let el = [[0.0, 0.0, 0.0]];
    assert!(cloud.synthesize_rf(&el, &[1.0], &{ let mut c = cfg(10); c.sound_speed = 0.0; c }).is_err());
    assert!(cloud.synthesize_rf(&el, &[1.0], &{ let mut c = cfg(10); c.sampling_frequency = -1.0; c }).is_err());
    assert!(cloud.synthesize_rf(&el, &[1.0], &cfg(0)).is_err());
    assert!(cloud.synthesize_rf(&el, &[], &cfg(10)).is_err());
    assert!(ScattererCloud::from_points(&[[0.0, 0.0, 0.0]], &[1.0, 2.0]).is_err());
}
