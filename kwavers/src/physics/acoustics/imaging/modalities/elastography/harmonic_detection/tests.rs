//! Tests for harmonic detection in elastography

use super::*;
use rustfft::num_complex::Complex;

#[test]
fn test_harmonic_detection_config() {
    let config = HarmonicDetectionConfig::default();
    assert_eq!(config.fundamental_frequency, 50.0);
    assert_eq!(config.n_harmonics, 3);
    assert_eq!(config.fft_window_size, 1024);
}

#[test]
fn test_harmonic_displacement_field_creation() {
    let field = HarmonicDisplacementField::new(10, 10, 10, 3, 100);

    assert_eq!(field.fundamental_magnitude.dim(), (10, 10, 10));
    assert_eq!(field.fundamental_phase.dim(), (10, 10, 10));
    assert_eq!(field.harmonic_magnitudes.len(), 3);
    assert_eq!(field.harmonic_phases.len(), 3);
    assert_eq!(field.harmonic_snrs.len(), 3);
    assert_eq!(field.time.len(), 100);
}

#[test]
fn test_harmonic_ratio_computation() {
    let mut field = HarmonicDisplacementField::new(5, 5, 5, 2, 50);

    // Set test values
    field.fundamental_magnitude.fill(1.0);
    field.harmonic_magnitudes[0].fill(0.1); // Second harmonic

    let ratio = field.harmonic_ratio(2);
    assert_eq!(ratio.dim(), (5, 5, 5));

    // Check ratio value
    for &val in ratio.iter() {
        assert!((val - 0.1).abs() < 1e-10);
    }
}

#[test]
fn test_window_function() {
    let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());
    let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let windowed = detector.apply_window(&time_series);

    assert_eq!(windowed.len(), time_series.len());
    // First and last values should be zero (Hann window)
    assert!((windowed[0] - 0.0).abs() < 1e-10);
    assert!((windowed[4] - 0.0).abs() < 1e-10);
}

#[test]
fn test_snr_computation() {
    let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());

    // Create test spectrum with signal peak
    let mut spectrum = vec![Complex::new(0.1, 0.0); 100];
    spectrum[50] = Complex::new(1.0, 0.0); // Strong signal

    let snr = detector.compute_snr(&spectrum, 50);
    assert!(snr > 0.0); // Should have positive SNR
}

#[test]
fn test_harmonic_detector_creation() {
    let config = HarmonicDetectionConfig {
        fundamental_frequency: 100.0,
        n_harmonics: 5,
        ..Default::default()
    };

    let _detector = HarmonicDetector::new(config);
    // Test passes if no panic occurs
}
