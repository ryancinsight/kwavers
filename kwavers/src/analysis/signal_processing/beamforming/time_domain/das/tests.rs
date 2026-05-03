//! Tests for time-domain DAS beamforming.

use super::core::{delay_and_sum, DEFAULT_DELAY_REFERENCE};
use crate::analysis::signal_processing::beamforming::time_domain::delay_reference::DelayReference;
use ndarray::Array3;

#[test]
fn das_aligns_impulses_with_reference_sensor_0() {
    let fs = 10.0;
    let n = 8usize;

    let delays = vec![1.0, 1.2];
    let weights = vec![1.0, 1.0];

    let mut x = Array3::<f64>::zeros((2, 1, n));
    x[[0, 0, 3]] = 1.0;
    x[[1, 0, 5]] = 1.0;

    let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect("das should succeed");

    assert!(
        (y[[0, 0, 3]] - 2.0).abs() < 1e-12,
        "both impulses should sum at t=3, got {}",
        y[[0, 0, 3]]
    );
}

#[test]
fn das_handles_earliest_arrival_reference() {
    let fs = 10.0;
    let n = 8usize;

    let delays = vec![1.0, 1.1, 0.9];
    let weights = vec![1.0, 1.0, 1.0];

    let mut x = Array3::<f64>::zeros((3, 1, n));
    x[[0, 0, 3]] = 1.0;
    x[[1, 0, 4]] = 1.0;
    x[[2, 0, 2]] = 1.0;

    let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::EarliestArrival)
        .expect("das should succeed");

    assert!(
        (y[[0, 0, 2]] - 3.0).abs() < 1e-12,
        "all impulses should sum at t=2 with EarliestArrival, got {}",
        y[[0, 0, 2]]
    );
}

#[test]
fn das_handles_negative_relative_delays() {
    let fs = 10.0;
    let n = 10usize;

    let delays = vec![1.0, 0.8];
    let weights = vec![1.0, 1.0];

    let mut x = Array3::<f64>::zeros((2, 1, n));
    x[[0, 0, 5]] = 1.0;
    x[[1, 0, 3]] = 1.0;

    let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect("das should handle negative shifts");

    let sum: f64 = y.iter().sum();
    assert!(
        (sum - 2.0).abs() < 1e-12,
        "total energy should be preserved, got {sum}"
    );
}

#[test]
fn das_applies_apodization_weights() {
    let fs = 10.0;
    let n = 8usize;

    let delays = vec![1.0, 1.0];
    let weights = vec![0.5, 1.5];

    let mut x = Array3::<f64>::zeros((2, 1, n));
    x[[0, 0, 3]] = 2.0;
    x[[1, 0, 3]] = 2.0;

    let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect("das should succeed");

    assert!(
        (y[[0, 0, 3]] - 4.0).abs() < 1e-12,
        "weighted sum should be 4.0, got {}",
        y[[0, 0, 3]]
    );
}

#[test]
fn das_rejects_invalid_channel_dimension() {
    let x = Array3::<f64>::zeros((2, 2, 8));
    let fs = 10.0;
    let delays = vec![1.0, 1.0];
    let weights = vec![1.0, 1.0];

    let err = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect_err("should reject non-1 channel dimension");

    assert!(
        err.to_string().contains("channels"),
        "error should mention channels"
    );
}

#[test]
fn das_rejects_empty_arrays() {
    let x = Array3::<f64>::zeros((0, 1, 8));
    let fs = 10.0;
    let delays: Vec<f64> = vec![];
    let weights: Vec<f64> = vec![];

    let err = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect_err("should reject empty arrays");

    assert!(err.to_string().contains("n_elements > 0"));
}

#[test]
fn das_rejects_invalid_sampling_frequency() {
    let x = Array3::<f64>::zeros((2, 1, 8));
    let delays = vec![1.0, 1.0];
    let weights = vec![1.0, 1.0];

    let err = delay_and_sum(&x, 0.0, &delays, &weights, DelayReference::SensorIndex(0))
        .expect_err("should reject fs=0");
    assert!(err.to_string().contains("finite"));

    let err = delay_and_sum(&x, -10.0, &delays, &weights, DelayReference::SensorIndex(0))
        .expect_err("should reject negative fs");
    assert!(err.to_string().contains("> 0"));

    let err = delay_and_sum(
        &x,
        f64::NAN,
        &delays,
        &weights,
        DelayReference::SensorIndex(0),
    )
    .expect_err("should reject NaN fs");
    assert!(err.to_string().contains("finite"));
}

#[test]
fn das_rejects_mismatched_array_lengths() {
    let x = Array3::<f64>::zeros((3, 1, 8));
    let fs = 10.0;
    let delays = vec![1.0, 1.0];
    let weights = vec![1.0, 1.0, 1.0];

    let err = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect_err("should reject mismatched delays length");
    assert!(err.to_string().contains("delays_s length"));

    let delays = vec![1.0, 1.0, 1.0];
    let weights = vec![1.0, 1.0];

    let err = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect_err("should reject mismatched weights length");
    assert!(err.to_string().contains("weights length"));
}

#[test]
fn das_preserves_zero_input() {
    let x = Array3::<f64>::zeros((2, 1, 8));
    let fs = 10.0;
    let delays = vec![1.0, 1.2];
    let weights = vec![1.0, 1.0];

    let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
        .expect("das should succeed");

    let sum: f64 = y.iter().sum();
    assert!(
        sum.abs() < 1e-14,
        "zero input should produce zero output, got sum={sum}"
    );
}

#[test]
fn default_delay_reference_is_sensor_zero() {
    assert_eq!(DEFAULT_DELAY_REFERENCE, DelayReference::SensorIndex(0));
}
