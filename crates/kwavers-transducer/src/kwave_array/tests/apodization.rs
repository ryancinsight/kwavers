//! Apodization window tests for [`KWaveArray`].

use super::super::{KWaveArray, KwaveApodizationWindow};

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
    let weights = array.get_apodization(KwaveApodizationWindow::Rectangular);
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
    let weights = array.get_apodization(KwaveApodizationWindow::Hann);
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
    let weights = array.get_apodization(KwaveApodizationWindow::Hamming);
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

/// Blackman window: endpoints ≈ 0, center = 1, symmetric.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_apodization_blackman_endpoints_and_symmetry() {
    let mut array = KWaveArray::new();
    for i in 0..9 {
        array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
    }
    let weights = array.get_apodization(KwaveApodizationWindow::Blackman);
    assert_eq!(weights.len(), 9);
    assert!(
        weights[0].abs() < 1e-12,
        "Blackman ends ~0, got {}",
        weights[0]
    );
    assert!(
        weights[8].abs() < 1e-12,
        "Blackman ends ~0, got {}",
        weights[8]
    );
    assert!((weights[4] - 1.0).abs() < 1e-12, "Blackman center = 1");
    for i in 0..9 {
        assert!(
            (weights[i] - weights[8 - i]).abs() < 1e-12,
            "Blackman not symmetric at i={i}"
        );
    }
}

/// Tukey window: `r = 0` ≡ rectangular, `r = 1` ≡ Hann, `0 < r < 1` has a flat
/// interior with tapered cosine edges.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_apodization_tukey_limits_and_flat_top() {
    let mut array = KWaveArray::new();
    for i in 0..9 {
        array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
    }
    // r = 0 ≡ rectangular.
    let rect = array.get_apodization(KwaveApodizationWindow::Rectangular);
    let t0 = array.get_apodization(KwaveApodizationWindow::Tukey(0.0));
    for (a, b) in t0.iter().zip(rect.iter()) {
        assert!((a - b).abs() < 1e-12, "Tukey(0) must equal rectangular");
    }
    // r = 1 ≡ Hann.
    let hann = array.get_apodization(KwaveApodizationWindow::Hann);
    let t1 = array.get_apodization(KwaveApodizationWindow::Tukey(1.0));
    for (a, b) in t1.iter().zip(hann.iter()) {
        assert!((a - b).abs() < 1e-12, "Tukey(1) must equal Hann");
    }
    // r = 0.5: tapered ends (0 at the edges), flat unity interior, symmetric.
    let t = array.get_apodization(KwaveApodizationWindow::Tukey(0.5));
    assert!(t[0].abs() < 1e-12, "Tukey(0.5) edge ~0, got {}", t[0]);
    assert!(t[8].abs() < 1e-12, "Tukey(0.5) edge ~0, got {}", t[8]);
    assert!((t[4] - 1.0).abs() < 1e-12, "Tukey(0.5) center = 1");
    for i in 0..9 {
        assert!(
            (t[i] - t[8 - i]).abs() < 1e-12,
            "Tukey not symmetric at i={i}"
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
        KwaveApodizationWindow::Rectangular,
        KwaveApodizationWindow::Hann,
        KwaveApodizationWindow::Hamming,
        KwaveApodizationWindow::Blackman,
        KwaveApodizationWindow::Tukey(0.5),
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
    use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
    let mut arr = KWaveArray::new();
    arr.add_disc_element((0.0, 0.0, 0.0), 0.001, None);
    let delays = arr.get_focus_delays((0.0, 0.0, 1.0));
    assert!(
        (delays[0] - 1.0 / SOUND_SPEED_TISSUE).abs() < 1e-10,
        "default sound speed must equal SOUND_SPEED_TISSUE"
    );
}
