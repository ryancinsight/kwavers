use ndarray::Array4;

use super::super::config::{NeuralBeamformingConfig, NeuralBeamformingMode};
use super::super::types::BeamformingFeedback;
use super::NeuralBeamformer;

fn create_test_rf_data() -> Array4<f32> {
    Array4::from_elem((1, 64, 1024, 1), 0.1)
}

#[test]
fn test_beamformer_creation() {
    let config = NeuralBeamformingConfig::default();
    let beamformer = NeuralBeamformer::new(config);
    assert!(beamformer.is_ok());
}

#[test]
fn test_beamformer_creation_invalid_config() {
    let config = NeuralBeamformingConfig {
        network_architecture: vec![],
        ..Default::default()
    };
    let beamformer = NeuralBeamformer::new(config);
    assert!(beamformer.is_err());
}

#[test]
fn test_process_hybrid() {
    let config = NeuralBeamformingConfig {
        mode: NeuralBeamformingMode::Hybrid,
        ..Default::default()
    };
    let mut beamformer = NeuralBeamformer::new(config).unwrap();

    let rf_data = create_test_rf_data();
    let angles = vec![0.0];
    let result = beamformer.process(&rf_data, &angles);

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.processing_mode, "Hybrid");
    assert!((0.0..=1.0).contains(&result.confidence));
}

#[test]
fn test_process_neural_only() {
    let config = NeuralBeamformingConfig {
        mode: NeuralBeamformingMode::NeuralOnly,
        ..Default::default()
    };
    let mut beamformer = NeuralBeamformer::new(config).unwrap();

    let rf_data = create_test_rf_data();
    let angles = vec![0.0];
    let result = beamformer.process(&rf_data, &angles);

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.processing_mode, "NeuralOnly");
}

#[test]
fn test_process_adaptive() {
    let config = NeuralBeamformingConfig {
        mode: NeuralBeamformingMode::Adaptive,
        ..Default::default()
    };
    let mut beamformer = NeuralBeamformer::new(config).unwrap();

    let rf_data = create_test_rf_data();
    let angles = vec![0.0];
    let result = beamformer.process(&rf_data, &angles);

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.processing_mode.starts_with("Adaptive"));
}

#[test]
fn test_signal_quality_assessment() {
    let config = NeuralBeamformingConfig::default();
    let beamformer = NeuralBeamformer::new(config).unwrap();

    let rf_data = create_test_rf_data();
    let quality = beamformer.assess_signal_quality(&rf_data);

    assert!(quality.is_ok());
    let quality = quality.unwrap();
    assert!((0.0..=1.0).contains(&quality));
}

#[test]
fn test_adaptation() {
    let config = NeuralBeamformingConfig::default();
    let mut beamformer = NeuralBeamformer::new(config).unwrap();

    let feedback = BeamformingFeedback {
        improvement: 0.05,
        error_gradient: 0.02,
        signal_quality: 0.85,
    };

    let result = beamformer.adapt(&feedback);
    assert!(result.is_ok());
}

#[test]
fn test_metrics_tracking() {
    let config = NeuralBeamformingConfig::default();
    let mut beamformer = NeuralBeamformer::new(config).unwrap();

    let rf_data = create_test_rf_data();
    let angles = vec![0.0];

    for _ in 0..3 {
        let _ = beamformer.process(&rf_data, &angles);
    }

    let metrics = beamformer.metrics();
    assert_eq!(metrics.total_frames_processed, 3);
    assert!(metrics.average_processing_time > 0.0);
}
