//! Apodization window tests for [`KWaveArray`].

use super::super::{ApodizationWindow, KWaveArray};

/// Rectangular apodization returns all-ones.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_apodization_rectangular_all_ones() {
    let mut array = KWaveArray::new();
    for i in 0..8 {
        array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
    }
    let weights = array.get_apodization(ApodizationWindow::Rectangular);
    assert_eq!(weights.len(), 8);
    for w in &weights {
        assert!((w - 1.0).abs() < 1e-15, "rectangular weight must be 1.0");
    }
}

/// Hann window: endpoints ≈ 0, center = 1.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_apodization_hann_endpoints_near_zero() {
    let mut array = KWaveArray::new();
    for i in 0..9 {
        array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
    }
    let weights = array.get_apodization(ApodizationWindow::Hann);
    assert_eq!(weights.len(), 9);
    assert!(
        weights[0].abs() < 1e-12,
        "Hann first weight must be ~0, got {}",
        weights[0]
    );
    assert!(
        weights[8].abs() < 1e-12,
        "Hann last weight must be ~0, got {}",
        weights[8]
    );
    assert!(
        (weights[4] - 1.0).abs() < 1e-12,
        "Hann center weight must be 1.0, got {}",
        weights[4]
    );
}

/// Hamming window: all weights in [0.08, 1.0] and symmetric.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_apodization_hamming_range_and_symmetry() {
    let mut array = KWaveArray::new();
    for i in 0..7 {
        array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
    }
    let weights = array.get_apodization(ApodizationWindow::Hamming);
    assert_eq!(weights.len(), 7);
    for &w in &weights {
        assert!(
            (0.07..=1.01).contains(&w),
            "Hamming weight out of range: {w}"
        );
    }
    for i in 0..7 {
        assert!(
            (weights[i] - weights[6 - i]).abs() < 1e-12,
            "Hamming not symmetric at i={i}: w[{i}]={} w[{}]={}",
            weights[i],
            6 - i,
            weights[6 - i]
        );
    }
}

/// Single-element array: all windows return `[1.0]`.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_apodization_single_element() {
    let mut array = KWaveArray::new();
    array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
    for window in [
        ApodizationWindow::Rectangular,
        ApodizationWindow::Hann,
        ApodizationWindow::Hamming,
    ] {
        let weights = array.get_apodization(window);
        assert_eq!(weights.len(), 1);
        assert!(
            (weights[0] - 1.0).abs() < 1e-12,
            "{window:?}: single element weight must be 1.0"
        );
    }
}

/// SSOT: `KWaveArray::new()` uses `SOUND_SPEED_TISSUE`.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_default_sound_speed_is_ssot_constant() {
    use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
    let mut arr = KWaveArray::new();
    arr.add_disc_element((0.0, 0.0, 0.0), 0.001, None);
    let delays = arr.get_focus_delays((0.0, 0.0, 1.0));
    assert!(
        (delays[0] - 1.0 / SOUND_SPEED_TISSUE).abs() < 1e-10,
        "default sound speed must equal SOUND_SPEED_TISSUE"
    );
}
