//! Tests for `PolynomialFilter` and `PolynomialFilterConfig`.

use ndarray::Array2;

use super::config::PolynomialFilterConfig;
use super::filter::PolynomialFilter;
use crate::core::constants::numerical::TWO_PI;

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
            let blood = 0.5 * (TWO_PI * (t as f64) / 10.0).sin();
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
            let blood = 2.0 * (TWO_PI * (t as f64) / 5.0).sin();
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

// ─── Exact value-semantic tests (normal equations derived) ────────────────────

/// Order-1 filter on a pure linear signal [0,1,2,3] yields exact zero residuals.
///
/// Normal equations for V=[[1,0],[1,1],[1,2],[1,3]], x=[0,1,2,3]:
///   VᵀV = [[4,6],[6,14]], Vᵀx = [6,14], det=20.
///   a = [(14·6−6·14)/20, (4·14−6·6)/20] = [0, 1].
///   fit = 0+t = x → residual = 0 for all t.
#[test]
fn polynomial_filter_linear_signal_yields_zero_residual() {
    let config = PolynomialFilterConfig {
        polynomial_order: 1,
        normalize_time: false,
    };
    let filter = PolynomialFilter::new(config).unwrap();
    let data = Array2::<f64>::from_shape_vec((1, 4), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let result = filter.filter(&data).unwrap();
    for t in 0..4 {
        assert!(
            result[[0, t]].abs() < 1e-12,
            "residual[0,{t}] = {} (expected 0.0 for linear signal)",
            result[[0, t]]
        );
    }
}

/// Order-1 filter on a constant signal [5,5,5,5] yields exact zero residuals.
///
/// Vᵀx = [20,30], a = [(14·20−6·30)/20, (4·30−6·20)/20] = [5, 0].
/// fit = 5+0t = 5 → residual = 0 for all t.
#[test]
fn polynomial_filter_constant_signal_yields_zero_residual() {
    let config = PolynomialFilterConfig {
        polynomial_order: 1,
        normalize_time: false,
    };
    let filter = PolynomialFilter::new(config).unwrap();
    let data = Array2::<f64>::from_shape_vec((1, 4), vec![5.0, 5.0, 5.0, 5.0]).unwrap();
    let result = filter.filter(&data).unwrap();
    for t in 0..4 {
        assert!(
            result[[0, t]].abs() < 1e-12,
            "residual[0,{t}] = {} (expected 0.0 for constant signal)",
            result[[0, t]]
        );
    }
}

/// Order-1 filter on quadratic [0,1,4,9] leaves the non-linear residual [1,−1,−1,1].
///
/// Vᵀx = [14,36], a = [(14·14−6·36)/20, (4·36−6·14)/20] = [−1, 3].
/// fit = −1+3t = [−1,2,5,8].
/// residual = x−fit = [1,−1,−1,1].
#[test]
fn polynomial_filter_order1_quadratic_signal_exact_residuals() {
    let config = PolynomialFilterConfig {
        polynomial_order: 1,
        normalize_time: false,
    };
    let filter = PolynomialFilter::new(config).unwrap();
    let data = Array2::<f64>::from_shape_vec((1, 4), vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let result = filter.filter(&data).unwrap();
    let expected = [1.0, -1.0, -1.0, 1.0];
    for (t, &exp) in expected.iter().enumerate() {
        assert!(
            (result[[0, t]] - exp).abs() < 1e-12,
            "residual[0,{t}] = {} (expected {exp})",
            result[[0, t]]
        );
    }
}

/// Order-1 filter with normalize_time=true on [0,1,2,3] still yields zero residual.
///
/// t_norm = [0, 1/3, 2/3, 1]; signal = 3·t_norm.
/// Normal equations give a = [0, 3]; fit = 3·t_norm = [0,1,2,3] → residual = 0.
#[test]
fn polynomial_filter_linear_signal_normalized_time_zero_residual() {
    let config = PolynomialFilterConfig {
        polynomial_order: 1,
        normalize_time: true,
    };
    let filter = PolynomialFilter::new(config).unwrap();
    let data = Array2::<f64>::from_shape_vec((1, 4), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let result = filter.filter(&data).unwrap();
    for t in 0..4 {
        assert!(
            result[[0, t]].abs() < 1e-12,
            "normalized_time residual[0,{t}] = {} (expected 0.0)",
            result[[0, t]]
        );
    }
}

/// Two pixels with different polynomial trends are each filtered independently.
///
/// pixel 0: [0,1,2,3] (linear) → residual [0,0,0,0].
/// pixel 1: [5,5,5,5] (constant) → residual [0,0,0,0].
#[test]
fn polynomial_filter_two_pixels_independent_filtering() {
    let config = PolynomialFilterConfig {
        polynomial_order: 1,
        normalize_time: false,
    };
    let filter = PolynomialFilter::new(config).unwrap();
    let data = Array2::<f64>::from_shape_vec((2, 4), vec![0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0])
        .unwrap();
    let result = filter.filter(&data).unwrap();
    for pixel in 0..2 {
        for t in 0..4 {
            assert!(
                result[[pixel, t]].abs() < 1e-12,
                "pixel={pixel} residual[{t}] = {} (expected 0.0)",
                result[[pixel, t]]
            );
        }
    }
}
