use ndarray::Array4;

use crate::core::error::KwaversResult;

use super::super::types::AIBeamformingConfig;
use super::processor::AIEnhancedBeamformingProcessor;
use super::trait_engine::PinnInferenceEngine;

#[derive(Debug)]
struct TestPinnEngine;

impl PinnInferenceEngine for TestPinnEngine {
    fn predict_realtime(
        &mut self,
        x_coords: &[f32],
        y_coords: &[f32],
        t_coords: &[f32],
    ) -> KwaversResult<(Vec<f32>, Vec<f32>)> {
        let n = x_coords.len().min(y_coords.len()).min(t_coords.len());
        Ok((vec![0.0; n], vec![0.25; n]))
    }
}

#[test]
fn test_processor_creation() {
    let config = AIBeamformingConfig {
        enable_realtime_pinn: false,
        ..Default::default()
    };
    let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];

    let _processor = AIEnhancedBeamformingProcessor::new(config, sensor_positions, None).unwrap();
}

#[test]
fn test_memory_estimation() {
    let config = AIBeamformingConfig {
        enable_realtime_pinn: false,
        ..Default::default()
    };
    let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
    let processor = AIEnhancedBeamformingProcessor::new(config, sensor_positions, None).unwrap();

    let memory_mb = processor.estimate_memory_usage();
    assert!(memory_mb > 0.0);
    assert!(memory_mb < 1000.0);
}

#[test]
fn test_config_access() {
    let config = AIBeamformingConfig {
        enable_realtime_pinn: false,
        ..Default::default()
    };
    let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
    let processor = AIEnhancedBeamformingProcessor::new(config, sensor_positions, None).unwrap();

    let retrieved_config = processor.config();
    assert_eq!(retrieved_config.performance_target_ms, 100.0);
}

#[test]
fn test_beamforming() {
    let config = AIBeamformingConfig {
        enable_realtime_pinn: false,
        ..Default::default()
    };
    let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
    let processor = AIEnhancedBeamformingProcessor::new(config, sensor_positions, None).unwrap();

    let rf_data = Array4::<f32>::from_elem((100, 10, 20, 1), 1.0);
    let angles = vec![0.0; 20];

    let volume = processor
        .perform_beamforming(rf_data.view(), &angles)
        .unwrap();
    assert_eq!(volume.dim(), (64, 64, 20));
}

#[test]
fn test_full_pipeline() {
    let config = AIBeamformingConfig {
        enable_realtime_pinn: true,
        ..Default::default()
    };
    let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
    let mut processor = AIEnhancedBeamformingProcessor::new(
        config,
        sensor_positions,
        Some(Box::new(TestPinnEngine)),
    )
    .unwrap();

    let rf_data = Array4::<f32>::from_elem((100, 10, 20, 1), 0.5);
    let angles = vec![0.0; 20];

    let ai_result = processor
        .process_ai_enhanced(rf_data.view(), &angles)
        .unwrap();
    assert_eq!(ai_result.volume.dim(), (64, 64, 20));
    assert!(ai_result.performance.total_time_ms > 0.0);
    assert!(!ai_result.features.is_empty());
}
