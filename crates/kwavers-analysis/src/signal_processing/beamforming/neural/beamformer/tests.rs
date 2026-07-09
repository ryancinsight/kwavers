use leto::Array4;

use super::super::config::{NeuralBeamformingConfig, NeuralBeamformingMode};
use super::super::types::BeamformingFeedback;
use super::NeuralBeamformer;

fn create_test_rf_data() -> Array4<f32> {
    Array4::from_elem((1, 64, 1024, 1), 0.1)
}

#[test]
fn test_beamformer_creation() {
    // default: mode=Hybrid, neural_network=Some (Hybrid allocates network)
    let config = NeuralBeamformingConfig::default();
    let beamformer = NeuralBeamformer::new(config).unwrap();
    assert!(
        beamformer.neural_network.is_some(),
        "Hybrid mode must allocate a neural network"
    );
}

#[test]
fn test_beamformer_creation_invalid_config() {
    let config = NeuralBeamformingConfig {
        network_architecture: vec![],
        ..Default::default()
    };
    let err = NeuralBeamformer::new(config).unwrap_err();
    assert!(
        !format!("{err:?}").is_empty(),
        "empty network_architecture must produce a non-empty error"
    );
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
    let result = beamformer.process(&rf_data, &angles).unwrap();
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
    let result = beamformer.process(&rf_data, &angles).unwrap();
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
    let result = beamformer.process(&rf_data, &angles).unwrap();
    assert!(result.processing_mode.starts_with("Adaptive"));
}

#[test]
fn test_signal_quality_assessment() {
    let config = NeuralBeamformingConfig::default();
    let beamformer = NeuralBeamformer::new(config).unwrap();

    let rf_data = create_test_rf_data();
    let quality = beamformer.assess_signal_quality(&rf_data).unwrap();
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

    beamformer.adapt(&feedback).unwrap();
    // Verify the adaptation updated metrics or state (metrics.total_frames_processed unchanged)
    assert_eq!(beamformer.metrics().total_frames_processed, 0);
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
