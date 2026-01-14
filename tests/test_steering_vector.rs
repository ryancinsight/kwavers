use kwavers::domain::sensor::beamforming::sensor_beamformer::SensorBeamformer;
use kwavers::domain::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
use kwavers::domain::sensor::localization::Position;
use std::f64::consts::PI;

#[test]
fn test_calculate_steering() {
    // Parameters
    let sound_speed = 1500.0;
    let frequency = 1000.0; // 1 kHz
    let wavelength = sound_speed / frequency; // 1.5 m
    let d = wavelength / 2.0; // 0.75 m (half wavelength spacing)
    let sampling_freq = 10000.0;

    // Create 2-element linear array along X-axis
    let s1 = Sensor::new(0, Position::new(0.0, 0.0, 0.0));
    let s2 = Sensor::new(1, Position::new(d, 0.0, 0.0));
    let sensors = vec![s1, s2];

    let array = SensorArray::new(sensors, sound_speed, ArrayGeometry::Linear);
    let beamformer = SensorBeamformer::new(array, sampling_freq);

    // Test Angles: (theta, phi)
    // 1. Broadside: theta = 0 (Z-axis). Direction [0, 0, 1]. Path diff 0.
    // 2. Endfire: theta = PI/2, phi = 0 (X-axis). Direction [1, 0, 0]. Path diff d. Phase diff PI.

    // Note: The conversion in code is:
    // x = sin(theta) * cos(phi)
    // y = sin(theta) * sin(phi)
    // z = cos(theta)

    // So for Z-axis (broadside to X-array): theta = 0.
    // For X-axis (endfire): theta = PI/2, phi = 0.

    let angles = vec![
        (0.0, 0.0),      // Broadside (Z)
        (PI / 2.0, 0.0), // Endfire (X)
    ];

    // Call calculate_steering with NEW signature
    // This will fail to compile until I update the implementation.
    let result = beamformer
        .calculate_steering(&angles, frequency, sound_speed)
        .expect("Calculation failed");

    // Check dimensions: (n_sensors, n_angles) = (2, 2)
    assert_eq!(result.shape(), &[2, 2]);

    // Check Broadside (Column 0)
    // Expect equal phases (or 0 phase if reference is at origin)
    // Sensor 1 at origin, Sensor 2 at (d,0,0).
    // Wave from Z (0,0,1). Dot product r1.k = 0. r2.k = (d,0,0).(0,0,k) = 0.
    // So both should have phase 0 (exp(0) = 1).
    let b_s1 = result[[0, 0]];
    let b_s2 = result[[1, 0]];

    println!("Broadside: s1={}, s2={}", b_s1, b_s2);
    // Magnitudes should be 1.0
    assert!((b_s1.norm() - 1.0).abs() < 1e-6);
    assert!((b_s2.norm() - 1.0).abs() < 1e-6);
    // Phases should be equal (difference is 0)
    let phase_diff_broadside = (b_s2 * b_s1.conj()).arg();
    assert!(
        phase_diff_broadside.abs() < 1e-6,
        "Broadside phase diff: {}",
        phase_diff_broadside
    );

    // Check Endfire (Column 1)
    // Wave from X (1,0,0).
    // r1 = (0,0,0). Phase 0.
    // r2 = (d,0,0). Phase k*d.
    // k = 2*pi/lambda. d = lambda/2.
    // k*d = pi.
    // So phase diff should be pi.
    // Note: SteeringVector computes exp(j k r.u).
    // So for sensor 2: exp(j * k * d).
    let e_s1 = result[[0, 1]];
    let e_s2 = result[[1, 1]];

    println!("Endfire: s1={}, s2={}", e_s1, e_s2);
    assert!((e_s1.norm() - 1.0).abs() < 1e-6);
    assert!((e_s2.norm() - 1.0).abs() < 1e-6);

    let phase_diff_endfire = (e_s2 * e_s1.conj()).arg();
    // phase_diff should be close to PI or -PI
    assert!(
        (phase_diff_endfire.abs() - PI).abs() < 1e-6,
        "Endfire phase diff: {} (expected PI)",
        phase_diff_endfire
    );
}
