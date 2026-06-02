use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;

use super::normalize_weights;
use super::*;

#[test]
fn test_compute_delay() {
    let delay = compute_delay(
        32,                 // center element
        64,                 // 64-element array
        512,                // sample index
        0.0003,             // 300 µm pitch
        0.05,               // 50 mm focal depth
        SOUND_SPEED_TISSUE, // sound speed
        40e6,               // 40 MHz sampling
    )
    .unwrap();

    // Geometric delay should be ~focal_depth / sound_speed
    assert!(delay > 0.0);
    assert!(delay < 1.0); // Should be sub-second
}

#[test]
fn test_compute_weights() {
    let weights = compute_weights(
        64,                 // 64 elements
        512,                // sample index
        0.0003,             // 300 µm pitch
        0.05,               // 50 mm focal depth
        SOUND_SPEED_TISSUE, // m/s
        5e6,                // 5 MHz center frequency
    )
    .unwrap();

    assert_eq!(weights.len(), 64);
    assert!(weights.iter().all(|&w| w.is_finite()));

    // Check normalization
    let sum: f32 = weights.iter().map(|w| w.abs()).sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Weights not normalized: sum = {}",
        sum
    );
}

#[test]
fn test_normalize_weights_normal() {
    let mut weights = vec![1.0, 2.0, 3.0, 4.0];
    normalize_weights(&mut weights);

    let sum: f32 = weights.iter().map(|w| w.abs()).sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_normalize_weights_zero() {
    let mut weights = vec![0.0, 0.0, 0.0, 0.0];
    normalize_weights(&mut weights);

    // Should become uniform
    for &w in &weights {
        assert!((w - 0.25).abs() < 1e-6);
    }
}

#[test]
fn test_hanning_apodization() {
    let n = 64;

    // Edge elements should be ~0
    assert!(hanning_apodization(0, n) < 0.01);
    assert!(hanning_apodization(n - 1, n) < 0.01);

    // Center element should be ~1
    let center = hanning_apodization(n / 2, n);
    assert!((center - 1.0).abs() < 0.01);

    // Symmetry check
    for i in 0..n / 2 {
        let left = hanning_apodization(i, n);
        let right = hanning_apodization(n - 1 - i, n);
        assert!((left - right).abs() < 1e-6, "Not symmetric at {}", i);
    }
}

#[test]
fn test_weight_symmetry() {
    let weights = compute_weights(
        32,                 // 32 elements
        100,                // sample index
        0.0003,             // pitch
        0.04,               // focal depth
        SOUND_SPEED_TISSUE, // sound speed
        5e6,                // frequency
    )
    .unwrap();

    // Weights should be symmetric for on-axis focus
    for i in 0..16 {
        let left = weights[i];
        let right = weights[31 - i];
        assert!(
            (left - right).abs() < 1e-4,
            "Asymmetric at {}: {} vs {}",
            i,
            left,
            right
        );
    }
}

#[test]
fn test_delay_increases_with_distance() {
    let center_delay = compute_delay(32, 64, 0, 0.0003, 0.05, SOUND_SPEED_TISSUE, 40e6).unwrap();
    let edge_delay = compute_delay(0, 64, 0, 0.0003, 0.05, SOUND_SPEED_TISSUE, 40e6).unwrap();

    // Edge elements have longer paths to on-axis focus
    assert!(edge_delay > center_delay);
}
