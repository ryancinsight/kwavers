//! Tests for `PolynomialFilter` and `PolynomialFilterConfig`.

use ndarray::Array2;

use super::config::PolynomialFilterConfig;
use super::filter::PolynomialFilter;

#[test]
fn test_config_validation() {
    let config = PolynomialFilterConfig::with_order(2);
    config.validate().unwrap();

    let bad_config = PolynomialFilterConfig {
        polynomial_order: 0,
        ..Default::default()
    };
    assert!(bad_config.validate().is_err());

    let too_high_config = PolynomialFilterConfig {
        polynomial_order: 11,
        ..Default::default()
    };
    assert!(too_high_config.validate().is_err());
}

#[test]
fn test_polynomial_filter_creation() {
    let config = PolynomialFilterConfig::with_order(2);
    let _filter = PolynomialFilter::new(config).unwrap();
}

#[test]
fn test_filter_with_linear_trend() {
    let config = PolynomialFilterConfig::with_order(1);
    let filter = PolynomialFilter::new(config).unwrap();

    let n_pixels = 10;
    let n_frames = 50;
    let mut data = Array2::<f64>::zeros((n_pixels, n_frames));

    for i in 0..n_pixels {
        for t in 0..n_frames {
            let trend = 10.0 * (t as f64) / (n_frames as f64);
            let blood = 0.5 * (2.0 * std::f64::consts::PI * (t as f64) / 10.0).sin();
            data[[i, t]] = trend + blood;
        }
    }

    let filtered = filter.filter(&data).unwrap();

    let original_mean = data.mean().unwrap();
    let filtered_mean = filtered.mean().unwrap().abs();

    assert!(filtered_mean < original_mean);
}

#[test]
fn test_filter_preserves_oscillations() {
    let config = PolynomialFilterConfig::with_order(2);
    let filter = PolynomialFilter::new(config).unwrap();

    let n_pixels = 5;
    let n_frames = 100;
    let mut data = Array2::<f64>::zeros((n_pixels, n_frames));

    for i in 0..n_pixels {
        for t in 0..n_frames {
            let t_norm = (t as f64) / (n_frames as f64);
            let trend = 10.0 * t_norm * t_norm;
            let blood = 2.0 * (2.0 * std::f64::consts::PI * (t as f64) / 5.0).sin();
            data[[i, t]] = trend + blood;
        }
    }

    let filtered = filter.filter(&data).unwrap();

    let filtered_std = filtered.std(0.0);
    assert!(filtered_std > 1.0);
}

#[test]
fn test_insufficient_frames() {
    let config = PolynomialFilterConfig::with_order(5);
    let filter = PolynomialFilter::new(config).unwrap();

    // 5 frames, order-5 polynomial requires > 5 frames.
    let data = Array2::<f64>::zeros((10, 5));
    let result = filter.filter(&data);

    assert!(result.is_err());
}
