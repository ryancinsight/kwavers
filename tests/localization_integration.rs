//! Integration tests for source localization algorithms
//!
//! This test suite demonstrates the usage of multilateration algorithms
//! for acoustic source detection.
//!
//! NOTE: MUSIC algorithm tests removed pending full implementation.
//! MUSIC currently has placeholder implementation in LocalizationProcessor::localize().
//! See backlog.md Sprint 213 Phase 1 for MUSIC implementation requirements:
//! - Complex Hermitian eigendecomposition (12-16 hours)
//! - AIC/MDL source counting
//! - 3D grid search and peak detection
//!
//! Once MUSIC is implemented, integration tests can be added following the
//! pattern of these multilateration tests.

use approx::assert_relative_eq;
use kwavers::analysis::signal_processing::localization::{Multilateration, MultilaterationConfig};

#[test]
fn test_multilateration_vs_trilateration() {
    // Compare multilateration (overdetermined) vs trilateration (minimal sensors)
    let c = 1500.0;

    // 6 sensors for overdetermined system
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
        [0.01, 0.01, 0.0],
        [0.01, 0.0, 0.01],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        max_iterations: 100,
        ..Default::default()
    };

    let multi = Multilateration::new(sensors.clone(), config).unwrap();

    // True source position
    let source_pos = [0.008, 0.008, 0.008];

    // Generate synthetic arrival times
    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist / c
        })
        .collect();

    let result = multi.localize(&arrival_times).unwrap();

    assert!(result.converged);
    assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-4);
    assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-4);
    assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-4);

    // GDOP should be reasonable for this geometry
    let gdop = multi.calculate_gdop(&result.position).unwrap();
    assert!(gdop > 0.0 && gdop < 20.0, "GDOP = {}", gdop);
}

#[test]
fn test_weighted_multilateration_with_heterogeneous_sensors() {
    let c = 1540.0;

    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.015, 0.0, 0.0],
        [0.0, 0.015, 0.0],
        [0.0, 0.0, 0.015],
        [0.01, 0.01, 0.01],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        use_weighted_ls: true,
        max_iterations: 100,
        ..Default::default()
    };

    let mut multi = Multilateration::new(sensors.clone(), config).unwrap();

    // Different sensor qualities (timing uncertainties)
    // Sensor 0 and 1 are high-precision, others are standard
    let uncertainties = vec![1e-10, 1e-10, 5e-10, 5e-10, 5e-10];
    multi.set_sensor_uncertainties(uncertainties).unwrap();

    let source_pos = [0.005, 0.005, 0.005];

    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist / c
        })
        .collect();

    let result = multi.localize(&arrival_times).unwrap();

    assert!(result.converged);
    assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-3);
    assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-3);
    assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-3);
}

#[test]
fn test_multilateration_poor_geometry() {
    // Test with collinear sensors (degenerate geometry)
    // This should fail with an error since the geometry matrix is not invertible
    let c = 1500.0;

    // All sensors along x-axis (degenerate geometry)
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.03, 0.0, 0.0],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        max_iterations: 100,
        ..Default::default()
    };

    let multi = Multilateration::new(sensors.clone(), config).unwrap();

    // Source offset from sensor line
    let source_pos = [0.015, 0.005, 0.005];

    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist / c
        })
        .collect();

    // Collinear sensors should fail with degenerate geometry error
    let result = multi.localize(&arrival_times);
    assert!(
        result.is_err(),
        "Expected error for degenerate collinear geometry"
    );

    // Error message should mention degenerate geometry or matrix not invertible
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("degenerate") || error_msg.contains("invertible"),
        "Expected degenerate geometry error, got: {}",
        error_msg
    );
}

#[test]
fn test_multilateration_noise_robustness() {
    // Test robustness to timing noise
    let c = 1500.0;

    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
        [0.01, 0.01, 0.0],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        max_iterations: 100,
        convergence_tolerance: 1e-6,
        ..Default::default()
    };

    let multi = Multilateration::new(sensors.clone(), config).unwrap();

    let source_pos = [0.01, 0.01, 0.01];

    // Add small timing noise (±0.1 sample at 40MHz = ±2.5ns)
    let noise = vec![2.5e-9, -2.5e-9, 1.5e-9, -1.5e-9, 0.5e-9];

    let arrival_times: Vec<f64> = sensors
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist / c + noise[i]
        })
        .collect();

    let result = multi.localize(&arrival_times).unwrap();

    assert!(result.converged);

    // With noise, expect degraded but reasonable accuracy
    let error = [
        (result.position[0] - source_pos[0]).abs(),
        (result.position[1] - source_pos[1]).abs(),
        (result.position[2] - source_pos[2]).abs(),
    ];

    // Error should be small (within wavelength at 1MHz: 1.5mm)
    let wavelength_1mhz = c / 1e6;
    assert!(
        error[0] < wavelength_1mhz,
        "X error {} > wavelength {}",
        error[0],
        wavelength_1mhz
    );
    assert!(
        error[1] < wavelength_1mhz,
        "Y error {} > wavelength {}",
        error[1],
        wavelength_1mhz
    );
    assert!(
        error[2] < wavelength_1mhz,
        "Z error {} > wavelength {}",
        error[2],
        wavelength_1mhz
    );
}

#[test]
fn test_multilateration_edge_cases() {
    let c = 1500.0;

    // Test with minimum sensor count (4 for 3D)
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        max_iterations: 100,
        ..Default::default()
    };

    let multi = Multilateration::new(sensors.clone(), config).unwrap();

    let source_pos = [0.01, 0.01, 0.01];

    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist / c
        })
        .collect();

    let result = multi.localize(&arrival_times).unwrap();

    assert!(result.converged);
    // Minimum sensor count should still work but may have lower accuracy
    assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-3);
    assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-3);
    assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-3);
}
