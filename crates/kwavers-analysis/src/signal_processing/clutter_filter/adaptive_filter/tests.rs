use super::filter::AdaptiveFilter;
use super::types::{AdaptiveFilterConfig, CbrEstimationMethod, SubspaceSeparationMethod};
use kwavers_core::constants::numerical::TWO_PI;
use ndarray::Array2;

#[test]
fn test_adaptive_filter_creation() {
    let config = AdaptiveFilterConfig::default();
    let _filter = AdaptiveFilter::new(config).unwrap();
}

#[test]
fn test_config_validation() {
    let config = AdaptiveFilterConfig {
        noise_floor_threshold: 1.5,
        ..Default::default()
    };
    assert!(AdaptiveFilter::new(config).is_err());

    let config = AdaptiveFilterConfig {
        temporal_smoothing: true,
        smoothing_window: 0,
        ..Default::default()
    };
    assert!(AdaptiveFilter::new(config).is_err());

    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 0 },
        ..Default::default()
    };
    assert!(AdaptiveFilter::new(config).is_err());
}

#[test]
fn test_filter_removes_low_frequency_component() {
    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 1 },
        cbr_estimation: CbrEstimationMethod::EigenvalueSum,
        noise_floor_threshold: 1e-6,
        temporal_smoothing: false,
        smoothing_window: 1,
    };

    let mut filter = AdaptiveFilter::new(config).unwrap();

    let n_frames = 16;
    let dc_component = 10.0;
    let mut data = Array2::<f64>::zeros((1, n_frames));
    for t in 0..n_frames {
        let oscillation = (TWO_PI * t as f64 / 4.0).cos();
        data[[0, t]] = dc_component + oscillation;
    }

    let filtered = filter.filter(&data).unwrap();
    let filtered_mean: f64 = filtered.row(0).mean().unwrap();
    assert!(filtered_mean.abs() < 0.5 * dc_component);
}

#[test]
fn test_adaptive_threshold_method() {
    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.2 },
        cbr_estimation: CbrEstimationMethod::EigenvalueSum,
        noise_floor_threshold: 1e-6,
        temporal_smoothing: false,
        smoothing_window: 1,
    };

    let mut filter = AdaptiveFilter::new(config).unwrap();

    let n_frames = 16;
    let mut data = Array2::<f64>::zeros((1, n_frames));
    for t in 0..n_frames {
        let low_freq = 5.0 * (TWO_PI * t as f64 / 16.0).cos();
        let high_freq = 1.0 * (TWO_PI * t as f64 / 2.0).cos();
        data[[0, t]] = low_freq + high_freq;
    }

    let filtered = filter.filter(&data).unwrap();
    assert!(filtered.iter().all(|&x| x.is_finite()));
    let cbr = filter.current_cbr_db().unwrap();
    assert!(cbr.is_finite());
}

#[test]
fn test_cbr_based_separation() {
    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::CbrBased {
            target_cbr_db: 20.0,
        },
        cbr_estimation: CbrEstimationMethod::PowerRatio,
        noise_floor_threshold: 1e-6,
        temporal_smoothing: false,
        smoothing_window: 1,
    };

    let mut filter = AdaptiveFilter::new(config).unwrap();

    let n_frames = 32;
    let mut data = Array2::<f64>::zeros((1, n_frames));
    for t in 0..n_frames {
        let clutter = 10.0 * (TWO_PI * t as f64 / 32.0).sin();
        let blood = 0.5 * (TWO_PI * t as f64 / 4.0).sin();
        data[[0, t]] = clutter + blood;
    }

    let _filtered = filter.filter(&data).unwrap();
    let cbr_db = filter.current_cbr_db().unwrap();
    assert!(cbr_db.is_finite());
    assert!(cbr_db > 0.0);
}

#[test]
fn test_filter_preserves_high_frequency() {
    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 2 },
        cbr_estimation: CbrEstimationMethod::EigenvalueSum,
        noise_floor_threshold: 1e-6,
        temporal_smoothing: false,
        smoothing_window: 1,
    };

    let mut filter = AdaptiveFilter::new(config).unwrap();

    let n_frames = 16;
    let mut data = Array2::<f64>::zeros((1, n_frames));
    for t in 0..n_frames {
        data[[0, t]] = (TWO_PI * t as f64 / 2.0).sin();
    }

    let original_power: f64 = data.iter().map(|&x| x * x).sum();
    let filtered = filter.filter(&data).unwrap();
    let filtered_power: f64 = filtered.iter().map(|&x| x * x).sum();
    assert!(filtered_power > 0.3 * original_power);
}

#[test]
fn test_insufficient_frames() {
    let config = AdaptiveFilterConfig::default();
    let mut filter = AdaptiveFilter::new(config).unwrap();
    let data = Array2::<f64>::zeros((1, 2));
    assert!(filter.filter(&data).is_err());
}

#[test]
fn test_cbr_history() {
    let config = AdaptiveFilterConfig::default();
    let mut filter = AdaptiveFilter::new(config).unwrap();

    let data = Array2::<f64>::from_shape_fn((3, 16), |(_, t)| (TWO_PI * t as f64 / 8.0).sin());

    filter.filter(&data).unwrap();
    assert_eq!(filter.cbr_history().len(), 3);

    filter.clear_history();
    assert_eq!(filter.cbr_history().len(), 0);
}
