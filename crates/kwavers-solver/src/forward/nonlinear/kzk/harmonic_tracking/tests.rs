use super::tracker::HarmonicTracker;
use super::types::{HarmonicAnalysis, HarmonicConfig};
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA, TWO_PI};
use leto::Array1;

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
    let mut pressure = Array1::zeros([n]);

    for i in 0..n {
        let phase = TWO_PI * config.frequency * (i as f64) * dt;
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
    let mut pressure = Array1::zeros([n]);

    for i in 0..n {
        let phase = TWO_PI * config.frequency * (i as f64) * dt;
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

    let distance_1mpa = tracker.predict_shock_distance(MPA_TO_PA);
    let distance_10mpa = tracker.predict_shock_distance(10.0 * MPA_TO_PA);

    assert!(distance_10mpa.unwrap() < distance_1mpa.unwrap());
}

#[test]
fn shock_distance_uses_fubini_frequency_dependent_contract() {
    let config = HarmonicConfig {
        frequency: MHZ_TO_HZ,
        b_a: 3.5,
        ..HarmonicConfig::default()
    };
    let tracker = HarmonicTracker::new(config);
    let pressure_amplitude = MPA_TO_PA;

    let rho0: f64 = DENSITY_WATER_NOMINAL;
    let c0: f64 = kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
    let beta = 1.0 + config.b_a / 2.0;
    let omega0 = TWO_PI * config.frequency;
    let expected = rho0 * c0.powi(3) / (beta * omega0 * pressure_amplitude);

    let distance = tracker.predict_shock_distance(pressure_amplitude).unwrap();

    assert!((distance - expected).abs() <= 1.0e-12 * expected);
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
    let empty_pressure = Array1::zeros([0]);

    let result = tracker.analyze_harmonics(&empty_pressure);
    assert!(result.is_err());
}
