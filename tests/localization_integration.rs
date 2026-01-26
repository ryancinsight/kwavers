//! Integration tests for source localization algorithms
//!
//! This test suite demonstrates the usage of multilateration and MUSIC
//! localization algorithms for acoustic source detection.

use approx::assert_relative_eq;
use kwavers::analysis::signal_processing::localization::{
    Multilateration, MultilaterationConfig, MusicConfig, MusicLocalizer,
};
use ndarray::Array2;
use num_complex::Complex;

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
fn test_music_single_source_detection() {
    let c = 1500.0;
    let freq = 500e3; // 500 kHz
    let wavelength = c / freq;

    // 5-element array
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [wavelength, 0.0, 0.0],
        [0.0, wavelength, 0.0],
        [0.0, 0.0, wavelength],
        [wavelength / 2.0, wavelength / 2.0, 0.0],
    ];

    let source_pos = [wavelength * 0.6, wavelength * 0.6, wavelength * 0.5];

    let config = MusicConfig {
        frequency: freq,
        sound_speed: c,
        num_sources: 1,
        x_bounds: [0.0, wavelength * 1.2],
        y_bounds: [0.0, wavelength * 1.2],
        z_bounds: [0.0, wavelength * 1.2],
        grid_resolution: wavelength / 8.0, // Coarse grid for test speed
        peak_separation: wavelength / 10.0,
        peak_threshold: 0.4,
    };

    let music = MusicLocalizer::new(sensors.clone(), config).unwrap();

    // Generate covariance matrix for single source
    let k = 2.0 * std::f64::consts::PI / wavelength;
    let steering_vec = music.compute_steering_vector(&source_pos, k);

    // R = a·a^H + σ²I
    let signal_power = 20.0;
    let noise_power = 0.5;

    let mut covariance: Array2<Complex<f64>> = Array2::from_elem((5, 5), Complex::new(0.0, 0.0));
    for i in 0..5 {
        for j in 0..5 {
            covariance[[i, j]] = steering_vec[i] * steering_vec[j].conj() * signal_power;
            if i == j {
                covariance[[i, j]] += Complex::new(noise_power, 0.0);
            }
        }
    }

    let result = music.localize(&covariance).unwrap();

    // Should detect exactly one source
    assert_eq!(result.sources.len(), 1);

    // Check eigenvalues: should have 1 large (signal) and 4 small (noise)
    assert_eq!(result.eigenvalues.len(), 5);
    assert!(result.eigenvalues[0] > 10.0); // Signal eigenvalue
    assert!(result.eigenvalues[4] < 5.0); // Noise eigenvalue

    // Detected position should be close (within grid resolution)
    let detected = result.sources[0];
    let error_x = (detected[0] - source_pos[0]).abs();
    let error_y = (detected[1] - source_pos[1]).abs();
    let error_z = (detected[2] - source_pos[2]).abs();

    // Allow error up to ~1.5 grid cells (due to discrete grid)
    let tolerance = wavelength / 4.0;
    assert!(error_x < tolerance, "X error: {} > {}", error_x, tolerance);
    assert!(error_y < tolerance, "Y error: {} > {}", error_y, tolerance);
    assert!(error_z < tolerance, "Z error: {} > {}", error_z, tolerance);
}

#[test]
fn test_music_covariance_estimation() {
    let c = 1500.0;
    let freq = 1e6;
    let wavelength = c / freq;

    let sensors = vec![
        [0.0, 0.0, 0.0],
        [wavelength / 2.0, 0.0, 0.0],
        [0.0, wavelength / 2.0, 0.0],
        [0.0, 0.0, wavelength / 2.0],
    ];

    let config = MusicConfig {
        frequency: freq,
        sound_speed: c,
        ..Default::default()
    };

    let music = MusicLocalizer::new(sensors, config).unwrap();

    // Generate snapshots with known statistics
    let num_snapshots = 200;
    let mut snapshots: Array2<Complex<f64>> =
        Array2::from_elem((4, num_snapshots), Complex::new(0.0, 0.0));

    // Single plane wave from specific direction
    let source_pos = [wavelength, wavelength, wavelength];
    let k = 2.0 * std::f64::consts::PI / wavelength;
    let steering_vec = music.compute_steering_vector(&source_pos, k);

    for t in 0..num_snapshots {
        let phase = 2.0 * std::f64::consts::PI * (t as f64) / (num_snapshots as f64);
        let signal = Complex::new(phase.cos(), phase.sin());

        for i in 0..4 {
            // Signal component
            snapshots[[i, t]] = steering_vec[i] * signal * 10.0;
            // Noise component (small)
            let noise_re = (i as f64) * 0.01;
            let noise_im = (t as f64) * 0.01;
            snapshots[[i, t]] += Complex::new(noise_re, noise_im);
        }
    }

    let covariance = music.estimate_covariance(&snapshots).unwrap();

    // Covariance should be Hermitian
    for i in 0..4 {
        for j in 0..4 {
            let cov_ij = covariance[[i, j]];
            let cov_ji = covariance[[j, i]];
            assert_relative_eq!(cov_ij.re, cov_ji.re, epsilon = 1e-10);
            assert_relative_eq!(cov_ij.im, -cov_ji.im, epsilon = 1e-10);
        }
    }

    // Diagonal should be real and positive
    for i in 0..4 {
        assert!(covariance[[i, i]].re > 0.0);
        assert_relative_eq!(covariance[[i, i]].im, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_complementary_localization_methods() {
    // Demonstrate using both methods for cross-validation
    let c = 1500.0;
    let freq = 750e3;
    let wavelength = c / freq;

    // Common sensor array
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [wavelength, 0.0, 0.0],
        [0.0, wavelength, 0.0],
        [0.0, 0.0, wavelength],
        [wavelength, wavelength, 0.0],
    ];

    let true_source = [wavelength * 0.5, wavelength * 0.5, wavelength * 0.4];

    // Method 1: Multilateration (time-based)
    let multi_config = MultilaterationConfig {
        sound_speed: c,
        ..Default::default()
    };
    let multi = Multilateration::new(sensors.clone(), multi_config).unwrap();

    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = true_source[0] - s[0];
            let dy = true_source[1] - s[1];
            let dz = true_source[2] - s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist / c
        })
        .collect();

    let multi_result = multi.localize(&arrival_times).unwrap();

    // Method 2: MUSIC (subspace-based)
    let music_config = MusicConfig {
        frequency: freq,
        sound_speed: c,
        num_sources: 1,
        x_bounds: [0.0, wavelength],
        y_bounds: [0.0, wavelength],
        z_bounds: [0.0, wavelength],
        grid_resolution: wavelength / 10.0,
        ..Default::default()
    };
    let music = MusicLocalizer::new(sensors.clone(), music_config).unwrap();

    let k = 2.0 * std::f64::consts::PI / wavelength;
    let steering_vec = music.compute_steering_vector(&true_source, k);

    let mut covariance: Array2<Complex<f64>> = Array2::from_elem((5, 5), Complex::new(0.0, 0.0));
    for i in 0..5 {
        for j in 0..5 {
            covariance[[i, j]] = steering_vec[i] * steering_vec[j].conj() * 15.0;
            if i == j {
                covariance[[i, j]] += Complex::new(0.5, 0.0);
            }
        }
    }

    let music_result = music.localize(&covariance).unwrap();

    // Both methods should converge
    assert!(multi_result.converged);
    assert!(!music_result.sources.is_empty());

    // Results should be consistent (within reasonable tolerance)
    let music_pos = music_result.sources[0];
    let multi_pos = multi_result.position;

    let difference = [
        (music_pos[0] - multi_pos[0]).abs(),
        (music_pos[1] - multi_pos[1]).abs(),
        (music_pos[2] - multi_pos[2]).abs(),
    ];

    // Allow for grid discretization in MUSIC (slightly more than grid resolution)
    let tolerance = wavelength / 4.0;
    assert!(
        difference[0] < tolerance,
        "X difference: {} > {}",
        difference[0],
        tolerance
    );
    assert!(
        difference[1] < tolerance,
        "Y difference: {} > {}",
        difference[1],
        tolerance
    );
    assert!(
        difference[2] < tolerance,
        "Z difference: {} > {}",
        difference[2],
        tolerance
    );
}
