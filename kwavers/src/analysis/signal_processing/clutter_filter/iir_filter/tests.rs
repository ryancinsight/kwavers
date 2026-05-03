use super::*;
use ndarray::Array2;

#[test]
fn test_config_validation() {
    let config = IirFilterConfig::with_cutoff(0.1);
    assert!(config.validate().is_ok());

    let bad_cutoff_low = IirFilterConfig {
        cutoff_frequency: 0.0,
        ..Default::default()
    };
    assert!(bad_cutoff_low.validate().is_err());

    let bad_cutoff_high = IirFilterConfig {
        cutoff_frequency: 0.6,
        ..Default::default()
    };
    assert!(bad_cutoff_high.validate().is_err());

    let bad_order = IirFilterConfig {
        cutoff_frequency: 0.1,
        order: 0,
        zero_phase: false,
    };
    assert!(bad_order.validate().is_err());
}

#[test]
fn test_iir_filter_creation() {
    let config = IirFilterConfig::with_cutoff(0.05);
    let filter = IirFilter::new(config);
    assert!(filter.is_ok());
}

#[test]
fn test_filter_removes_dc_component() {
    // Use higher cutoff for more aggressive DC removal
    let config = IirFilterConfig::with_cutoff(0.05).with_zero_phase();
    let filter = IirFilter::new(config).unwrap();

    // Create signal with DC offset + oscillation
    let n_pixels = 5;
    let n_frames = 100;
    let mut data = Array2::<f64>::zeros((n_pixels, n_frames));

    for i in 0..n_pixels {
        for t in 0..n_frames {
            let dc = 10.0; // DC component (clutter)
            let ac = 2.0 * (2.0 * std::f64::consts::PI * (t as f64) / 20.0).sin();
            data[[i, t]] = dc + ac;
        }
    }

    let filtered = filter.filter(&data).unwrap();

    // Check that DC component is significantly reduced
    let original_mean = data.mean().unwrap();
    let filtered_mean = filtered.mean().unwrap();

    // Filtered mean should be much smaller than original
    assert!(filtered_mean.abs() < 0.3 * original_mean.abs());
}

#[test]
fn test_filter_preserves_high_frequency() {
    let config = IirFilterConfig::with_cutoff(0.05);
    let filter = IirFilter::new(config).unwrap();

    // Create high-frequency oscillation
    let n_pixels = 3;
    let n_frames = 100;
    let mut data = Array2::<f64>::zeros((n_pixels, n_frames));

    for i in 0..n_pixels {
        for t in 0..n_frames {
            // High-frequency signal (blood flow)
            data[[i, t]] = 3.0 * (2.0 * std::f64::consts::PI * (t as f64) / 5.0).sin();
        }
    }

    let filtered = filter.filter(&data).unwrap();

    // Check that high-frequency content is largely preserved
    let filtered_std = filtered.std(0.0);
    let original_std = data.std(0.0);

    assert!(filtered_std > 0.8 * original_std); // Should retain most amplitude
}

#[test]
fn test_zero_phase_filtering() {
    let config = IirFilterConfig::with_cutoff(0.1).with_zero_phase();
    let filter = IirFilter::new(config).unwrap();

    let n_frames = 50;
    let signal =
        Array2::from_shape_fn((1, n_frames), |(_, t)| 5.0 + 2.0 * ((t as f64) * 0.4).sin());

    let filtered = filter.filter(&signal);
    assert!(filtered.is_ok());

    // Zero-phase filtering should preserve signal shape better
    // (this is a weak test - full test would compare phase spectrum)
    let result = filtered.unwrap();
    assert_eq!(result.dim(), signal.dim());
}

#[test]
fn test_higher_order_filter() {
    let config = IirFilterConfig::with_cutoff(0.05).with_order(2);
    let filter = IirFilter::new(config).unwrap();

    let data = Array2::from_shape_fn((5, 100), |(_, t)| 10.0 + (t as f64) * 0.1);

    let filtered = filter.filter(&data);
    assert!(filtered.is_ok());
}
