use super::tracker::HarmonicTracker;
use super::types::{HarmonicAnalysis, HarmonicConfig};
use ndarray::Array1;
use std::f64::consts::PI;

#[test]
fn test_harmonic_tracker_creation() {
    let config = HarmonicConfig::default();
    let _tracker = HarmonicTracker::new(config);
}

#[test]
fn test_pure_sinusoid_analysis() {
    let config = HarmonicConfig::default();
    let tracker = HarmonicTracker::new(config);

    let n = 1000;
    let dt = 1.0 / config.sampling_rate;
    let mut pressure = Array1::zeros(n);

    for i in 0..n {
        let phase = 2.0 * PI * config.frequency * (i as f64) * dt;
        pressure[i] = 100.0 * phase.sin();
    }

    let analysis = tracker.analyze_harmonics(&pressure).unwrap();
    assert!(analysis.rms_total > 0.0);
    assert!(analysis.crest_factor > 1.0);
    assert!(analysis.thd < 10.0);
}

#[test]
fn test_harmonic_content_distorted_wave() {
    let config = HarmonicConfig::default();
    let tracker = HarmonicTracker::new(config);

    let n = 1000;
    let dt = 1.0 / config.sampling_rate;
    let mut pressure = Array1::zeros(n);

    for i in 0..n {
        let phase = 2.0 * PI * config.frequency * (i as f64) * dt;
        pressure[i] = 100.0 * phase.sin() + 20.0 * (2.0 * phase).sin();
    }

    let analysis = tracker.analyze_harmonics(&pressure).unwrap();
    assert!(analysis.thd > 10.0);
    assert!(analysis.energy_ratio > 0.0);
}

#[test]
fn test_shock_distance_prediction() {
    let config = HarmonicConfig::default();
    let tracker = HarmonicTracker::new(config);

    let distance_1mpa = tracker.predict_shock_distance(1e6);
    let distance_10mpa = tracker.predict_shock_distance(10e6);

    assert!(distance_1mpa.is_some());
    assert!(distance_10mpa.is_some());
    assert!(distance_10mpa.unwrap() < distance_1mpa.unwrap());
}

#[test]
fn test_history_management() {
    let config = HarmonicConfig::default();
    let mut tracker = HarmonicTracker::new(config);

    let analysis = HarmonicAnalysis {
        thd: 5.0,
        ..Default::default()
    };

    tracker.record_analysis(analysis);
    assert_eq!(tracker.history().len(), 1);
    assert_eq!(tracker.history()[0].thd, 5.0);

    tracker.clear_history();
    assert_eq!(tracker.history().len(), 0);
}

#[test]
fn test_config_validation() {
    let config = HarmonicConfig::default();
    assert!(config.frequency > 0.0);
    assert!(config.max_harmonic > 0);
    assert!(config.b_a > 0.0);
}

#[test]
fn test_empty_pressure_handling() {
    let config = HarmonicConfig::default();
    let tracker = HarmonicTracker::new(config);
    let empty_pressure = Array1::zeros(0);

    let result = tracker.analyze_harmonics(&empty_pressure);
    assert!(result.is_err());
}
