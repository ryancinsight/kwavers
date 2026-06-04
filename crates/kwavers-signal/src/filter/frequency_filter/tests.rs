//! Tests for `FrequencyFilter`.

use super::filter::FrequencyFilter;
use crate::Filter;
use kwavers_core::constants::numerical::{TWO_PI};

fn sine_wave(freq: f64, sample_rate: f64, n_samples: usize) -> Vec<f64> {
    let dt = 1.0 / sample_rate;
    (0..n_samples)
        .map(|i| (TWO_PI * freq * i as f64 * dt).sin())
        .collect()
}

fn rms(signal: &[f64]) -> f64 {
    let sum_sq: f64 = signal.iter().map(|&x| x * x).sum();
    (sum_sq / signal.len() as f64).sqrt()
}

#[test]
fn test_bandpass_passes_in_band_frequency() {
    let filter = FrequencyFilter::new();
    let sample_rate = 10_000.0;
    let dt = 1.0 / sample_rate;
    let n = 1024;

    let signal = sine_wave(1000.0, sample_rate, n);
    let original_rms = rms(&signal);

    let filtered = filter.bandpass(&signal, dt, 500.0, 2000.0).unwrap();
    let filtered_rms = rms(&filtered);

    assert!(filtered_rms > 0.9 * original_rms);
}

#[test]
fn test_bandpass_rejects_out_of_band_frequency() {
    let filter = FrequencyFilter::new();
    let sample_rate = 10_000.0;
    let dt = 1.0 / sample_rate;
    let n = 1024;

    let signal = sine_wave(100.0, sample_rate, n);
    let original_rms = rms(&signal);

    let filtered = filter.bandpass(&signal, dt, 500.0, 2000.0).unwrap();
    let filtered_rms = rms(&filtered);

    assert!(filtered_rms < 0.1 * original_rms);
}

#[test]
fn test_lowpass_filters_high_frequencies() {
    let filter = FrequencyFilter::new();
    let sample_rate = 10_000.0;
    let dt = 1.0 / sample_rate;
    let n = 1024;

    let signal = sine_wave(3000.0, sample_rate, n);
    let filtered = filter.lowpass(&signal, dt, 2000.0).unwrap();
    let filtered_rms = rms(&filtered);

    assert!(filtered_rms < 0.1);
}

#[test]
fn test_highpass_filters_low_frequencies() {
    let filter = FrequencyFilter::new();
    let sample_rate = 10_000.0;
    let dt = 1.0 / sample_rate;
    let n = 1024;

    let signal = sine_wave(100.0, sample_rate, n);
    let filtered = filter.highpass(&signal, dt, 500.0).unwrap();
    let filtered_rms = rms(&filtered);

    assert!(filtered_rms < 0.1);
}

#[test]
fn test_time_window_zeros_outside_window() {
    let filter = FrequencyFilter::new();
    let dt = 0.0001;
    let signal = vec![1.0; 100];

    let windowed = filter
        .apply_time_window(signal, dt, (0.001, 0.003))
        .unwrap();

    assert!(
        windowed[0..10].iter().all(|&x| x == 0.0),
        "Samples before window should be zero"
    );
    assert!(
        windowed[10..=30].iter().all(|&x| x == 1.0),
        "Samples within window [0.001, 0.003] should be 1.0"
    );
    assert!(
        windowed[31..].iter().all(|&x| x == 0.0),
        "Samples after window should be zero"
    );
}

#[test]
fn test_filter_trait_implementation() {
    let filter = FrequencyFilter::new();
    let signal = vec![1.0, 2.0, 3.0];
    let result = filter.apply(&signal, 0.001).unwrap();
    assert_eq!(result, signal);
}

#[test]
fn test_empty_signal_handling() {
    let filter = FrequencyFilter::new();
    let empty: Vec<f64> = vec![];
    let result = filter.bandpass(&empty, 0.001, 100.0, 1000.0).unwrap();
    assert_eq!(result.len(), 0);
}
