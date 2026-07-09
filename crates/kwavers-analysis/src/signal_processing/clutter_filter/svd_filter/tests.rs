use super::*;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array2;

#[test]
fn test_config_validation() {
    let config = SvdClutterFilterConfig::with_fixed_rank(3);
    config.validate().unwrap();

    let bad_config = SvdClutterFilterConfig {
        clutter_rank: 0,
        ..Default::default()
    };
    assert!(bad_config.validate().is_err());
}

#[test]
fn test_svd_filter_creation() {
    let config = SvdClutterFilterConfig::with_fixed_rank(2);
    let _filter = SignalSvdClutterFilter::new(config).unwrap();
}

#[test]
fn test_filter_with_synthetic_data() {
    // Create synthetic data: clutter (low-rank) + blood (high-rank noise)
    let n_pixels = 100;
    let n_frames = 150;

    // Low-rank clutter component (tissue motion)
    let mut data = Array2::<f64>::zeros((n_pixels, n_frames));
    for i in 0..n_pixels {
        for t in 0..n_frames {
            // Low-frequency sinusoidal motion (clutter)
            let clutter = 10.0 * (TWO_PI * (t as f64) / 50.0).sin();
            // High-frequency noise (blood flow)
            let blood = 0.5 * ((i + t) as f64).sin();
            data[[i, t]] = clutter + blood;
        }
    }

    // Apply filter
    let config = SvdClutterFilterConfig::with_fixed_rank(2);
    let filter = SignalSvdClutterFilter::new(config).unwrap();
    let filtered = filter.filter(&data).unwrap();

    // Check that filtering reduced high-amplitude components
    let original_std = data.std(0.0);
    let filtered_std = filtered.std(0.0);
    assert!(filtered_std < original_std);
}

#[test]
fn test_auto_rank_selection() {
    let n_pixels = 50;
    let n_frames = 100;

    // Create data with clear rank structure
    let mut data = Array2::<f64>::zeros((n_pixels, n_frames));
    for i in 0..n_pixels {
        for t in 0..n_frames {
            // Strong clutter (low-rank)
            data[[i, t]] = 100.0 * (t as f64 / 10.0).sin() + 1.0 * ((i + t) as f64).sin();
        }
    }

    let config = SvdClutterFilterConfig::with_auto_rank(0.95);
    let filter = SignalSvdClutterFilter::new(config).unwrap();
    let filtered = filter.filter(&data).unwrap();

    assert_eq!(filtered.dim(), data.dim());
}

#[test]
fn test_power_doppler_computation() {
    let config = SvdClutterFilterConfig::default();
    let filter = SignalSvdClutterFilter::new(config).unwrap();

    // Create simple filtered data
    let n_pixels = 10;
    let n_frames = 100;
    let filtered =
        Array2::<f64>::from_shape_fn((n_pixels, n_frames), |(i, t)| ((i + t) as f64).sin());

    let power_doppler = filter.compute_power_doppler(&filtered);

    assert_eq!(power_doppler.len(), n_pixels);
    // All values should be positive (variance is always >= 0)
    assert!(power_doppler.iter().all(|&x| x >= 0.0));
}

#[test]
fn test_scr_improvement() {
    let config = SvdClutterFilterConfig::with_fixed_rank(1);
    let filter = SignalSvdClutterFilter::new(config).unwrap();

    let original = Array2::<f64>::from_elem((10, 50), 1.0);
    let filtered = Array2::<f64>::from_elem((10, 50), 0.1);

    let scr = filter
        .estimate_scr_improvement(&original, &filtered)
        .unwrap();

    // Should show improvement (positive dB)
    assert!(scr > 0.0);
}

#[test]
fn test_ensemble_length_validation() {
    let config = SvdClutterFilterConfig {
        clutter_rank: 5,
        min_ensemble_length: 100,
        ..Default::default()
    };
    let filter = SignalSvdClutterFilter::new(config).unwrap();

    // Too short ensemble
    let short_data = Array2::<f64>::zeros((10, 50));
    assert!(filter.filter(&short_data).is_err());

    // Sufficient ensemble
    let good_data = Array2::<f64>::zeros((10, 150));
    let filtered_good = filter.filter(&good_data).unwrap();
    assert_eq!(filtered_good.dim(), good_data.dim());
}
