//! Tests for harmonic detection in elastography

use super::spectral::hann_weight;
use super::*;
use apollo::Complex64;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, ValidationError};

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

    assert_eq!(field.fundamental_magnitude.shape(), [10, 10, 10]);
    assert_eq!(field.fundamental_phase.shape(), [10, 10, 10]);
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
    assert_eq!(ratio.shape(), [5, 5, 5]);

    // Check ratio value
    for &val in ratio.iter() {
        assert!((val - 0.1).abs() < 1e-10);
    }
}

#[test]
fn test_window_function() {
    let time_series = [1.0, 2.0, 3.0, 4.0, 5.0];

    let windowed: Vec<_> = time_series
        .iter()
        .enumerate()
        .map(|(index, &sample)| sample * hann_weight(index, time_series.len()))
        .collect();

    assert_eq!(windowed.len(), time_series.len());
    // First and last values should be zero (Hann window)
    assert!((windowed[0] - 0.0).abs() < 1e-10);
    assert!((windowed[4] - 0.0).abs() < 1e-10);
}

#[test]
fn test_snr_computation() {
    // Create test spectrum with signal peak
    let mut spectrum = vec![Complex64::new(0.1, 0.0); 100];
    spectrum[50] = Complex64::new(1.0, 0.0); // Strong signal

    let snr = HarmonicDetector::compute_snr(&spectrum, 50);
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

#[test]
fn detector_matches_direct_dft_for_distinct_points() {
    const SAMPLE_COUNT: usize = 128;
    const SAMPLING_FREQUENCY: f64 = 1024.0;
    const FUNDAMENTAL_FREQUENCY: f64 = 64.0;
    let config = HarmonicDetectionConfig {
        fundamental_frequency: FUNDAMENTAL_FREQUENCY,
        n_harmonics: 3,
        fft_window_size: SAMPLE_COUNT,
        ..Default::default()
    };
    let detector = HarmonicDetector::new(config);
    let mut samples = leto::Array4::zeros((2, 1, 1, SAMPLE_COUNT));
    for time_index in 0..SAMPLE_COUNT {
        let time = time_index as f64 / SAMPLING_FREQUENCY;
        samples[[0, 0, 0, time_index]] = (TWO_PI * FUNDAMENTAL_FREQUENCY * time).sin()
            + 0.25 * (2.0 * TWO_PI * FUNDAMENTAL_FREQUENCY * time).sin();
        samples[[1, 0, 0, time_index]] = 0.5 * (TWO_PI * FUNDAMENTAL_FREQUENCY * time + 0.4).sin()
            + 0.1 * (2.0 * TWO_PI * FUNDAMENTAL_FREQUENCY * time - 0.2).sin();
    }

    let field = detector
        .analyze_harmonics(&samples, SAMPLING_FREQUENCY)
        .expect("finite harmonic input");
    let fundamental_bin = 8;
    let second_harmonic_bin = 16;
    let third_harmonic_bin = 24;
    assert_eq!(field.harmonic_magnitudes.len(), 2);
    for point in 0..2 {
        let point_samples: Vec<_> = (0..SAMPLE_COUNT)
            .map(|time| samples[[point, 0, 0, time]])
            .collect();
        let reference = direct_dft(&point_samples);
        assert_polar_close(
            field.fundamental_magnitude[[point, 0, 0]],
            field.fundamental_phase[[point, 0, 0]],
            reference[fundamental_bin],
            &point_samples,
        );
        assert_polar_close(
            field.harmonic_magnitudes[0][[point, 0, 0]],
            field.harmonic_phases[0][[point, 0, 0]],
            reference[second_harmonic_bin],
            &point_samples,
        );
        assert_polar_close(
            field.harmonic_magnitudes[1][[point, 0, 0]],
            field.harmonic_phases[1][[point, 0, 0]],
            reference[third_harmonic_bin],
            &point_samples,
        );
    }
    assert_eq!(field.frequency[1], 8.0);
    assert_eq!(field.frequency[64], 512.0);
    assert_ne!(
        field.fundamental_magnitude[[0, 0, 0]],
        field.fundamental_magnitude[[1, 0, 0]]
    );
}

#[test]
fn detector_preserves_strided_logical_point_order() {
    const SAMPLE_COUNT: usize = 32;
    const SAMPLING_FREQUENCY: f64 = 256.0;
    const FUNDAMENTAL_FREQUENCY: f64 = 32.0;
    let layout =
        leto::Layout::f_contiguous([2, 1, 1, SAMPLE_COUNT]).expect("small Fortran layout is valid");
    let mut samples = leto::Array4::new(layout, leto::VecStorage::new(vec![0.0; 2 * SAMPLE_COUNT]))
        .expect("storage covers the strided layout");
    for point in 0..2 {
        for time_index in 0..SAMPLE_COUNT {
            let time = time_index as f64 / SAMPLING_FREQUENCY;
            samples[[point, 0, 0, time_index]] = (point + 1) as f64
                * (TWO_PI * FUNDAMENTAL_FREQUENCY * time + point as f64 * 0.2).sin();
        }
    }
    assert!(samples.as_slice().is_none());

    let detector = HarmonicDetector::new(HarmonicDetectionConfig {
        fundamental_frequency: FUNDAMENTAL_FREQUENCY,
        n_harmonics: 2,
        ..Default::default()
    });
    let field = detector
        .analyze_harmonics(&samples, SAMPLING_FREQUENCY)
        .expect("finite strided harmonic input");
    for point in 0..2 {
        let point_samples: Vec<_> = (0..SAMPLE_COUNT)
            .map(|time| samples[[point, 0, 0, time]])
            .collect();
        let reference = direct_dft(&point_samples);
        assert_polar_close(
            field.fundamental_magnitude[[point, 0, 0]],
            field.fundamental_phase[[point, 0, 0]],
            reference[4],
            &point_samples,
        );
    }
}

#[test]
fn detector_rejects_invalid_spectral_domains() {
    let valid_samples = leto::Array4::zeros((1, 1, 1, 8));
    let empty_samples = leto::Array4::zeros((1, 1, 1, 0));
    let short_samples = leto::Array4::zeros((1, 1, 1, 1));
    let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());
    for samples in [&empty_samples, &short_samples] {
        assert_invalid_parameter(
            detector.analyze_harmonics(samples, 1_000.0),
            "displacement_time_series time extent",
        );
    }
    for sampling_frequency in [0.0, f64::NAN, f64::INFINITY] {
        assert_invalid_parameter(
            detector.analyze_harmonics(&valid_samples, sampling_frequency),
            "sampling_frequency",
        );
    }

    let no_harmonics = HarmonicDetector::new(HarmonicDetectionConfig {
        n_harmonics: 0,
        ..Default::default()
    });
    assert_invalid_parameter(
        no_harmonics.analyze_harmonics(&valid_samples, 1_000.0),
        "HarmonicDetectionConfig.n_harmonics",
    );

    for fundamental_frequency in [0.0, f64::NAN, f64::INFINITY] {
        let invalid_fundamental = HarmonicDetector::new(HarmonicDetectionConfig {
            fundamental_frequency,
            ..Default::default()
        });
        assert_invalid_parameter(
            invalid_fundamental.analyze_harmonics(&valid_samples, 1_000.0),
            "HarmonicDetectionConfig.fundamental_frequency",
        );
    }

    let aliased_fundamental = HarmonicDetector::new(HarmonicDetectionConfig {
        fundamental_frequency: 501.0,
        n_harmonics: 1,
        ..Default::default()
    });
    assert_invalid_parameter(
        aliased_fundamental.analyze_harmonics(&valid_samples, 1_000.0),
        "HarmonicDetectionConfig.fundamental_frequency",
    );
    let aliased_harmonics = HarmonicDetector::new(HarmonicDetectionConfig {
        fundamental_frequency: 200.0,
        n_harmonics: 3,
        ..Default::default()
    });
    assert_invalid_parameter(
        aliased_harmonics.analyze_harmonics(&valid_samples, 1_000.0),
        "HarmonicDetectionConfig.n_harmonics",
    );
}

fn direct_dft(samples: &[f64]) -> Vec<Complex64> {
    let windowed: Vec<_> = samples
        .iter()
        .enumerate()
        .map(|(index, &sample)| sample * hann_weight(index, samples.len()))
        .collect();
    let normalization = (samples.len() as f64).sqrt();
    (0..samples.len())
        .map(|frequency| {
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (time, &sample) in windowed.iter().enumerate() {
                let angle = -TWO_PI * frequency as f64 * time as f64 / samples.len() as f64;
                real += sample * angle.cos();
                imaginary += sample * angle.sin();
            }
            Complex64::new(real / normalization, imaginary / normalization)
        })
        .collect()
}

fn assert_polar_close(
    actual_magnitude: f64,
    actual_phase: f64,
    expected: Complex64,
    samples: &[f64],
) {
    let normalized_l1 = samples
        .iter()
        .enumerate()
        .map(|(index, sample)| (sample * hann_weight(index, samples.len())).abs())
        .sum::<f64>()
        / (samples.len() as f64).sqrt();
    // Each direct-DFT component performs O(N) rounded transcendental and
    // multiply-add operations. The factor 128 conservatively bounds their
    // accumulated error relative to the normalized input L1 norm.
    let magnitude_bound = 128.0 * samples.len() as f64 * f64::EPSILON * normalized_l1.max(1.0);
    let phase_bound = 2.0 * magnitude_bound / expected.norm().max(magnitude_bound);
    assert!((actual_magnitude - expected.norm()).abs() <= magnitude_bound);
    assert!(wrapped_phase_distance(actual_phase, expected.arg()) <= phase_bound);
}

fn wrapped_phase_distance(left: f64, right: f64) -> f64 {
    let distance = (left - right).rem_euclid(TWO_PI);
    distance.min(TWO_PI - distance)
}

fn assert_invalid_parameter(
    result: Result<HarmonicDisplacementField, KwaversError>,
    expected_parameter: &str,
) {
    match result {
        Err(KwaversError::Validation(ValidationError::InvalidValue { parameter, .. })) => {
            assert_eq!(parameter, expected_parameter);
        }
        other => panic!("expected InvalidValue for {expected_parameter}, got {other:?}"),
    }
}
