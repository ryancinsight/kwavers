use super::core::NeuralBeamformingProcessor;
use crate::analysis::signal_processing::beamforming::neural::types::PINNBeamformingConfig;

#[test]
fn test_processor_creation() {
    let config = PINNBeamformingConfig::default();
    let processor = NeuralBeamformingProcessor::new(config);
    assert!(processor.is_ok());
}

#[test]
fn test_memory_calculation() {
    let config = PINNBeamformingConfig::default();
    let processor = NeuralBeamformingProcessor::new(config).unwrap();
    let memory = processor.calculate_memory_requirement();
    assert!(memory > 0);
}

#[cfg(feature = "pinn")]
#[test]
fn test_process_volume() {
    use ndarray::Array4;

    let config = PINNBeamformingConfig {
        rf_data_channels: 8,
        samples_per_channel: 128,
        volume_size: (2, 8, 128),
        enable_pinn: false,
        enable_uncertainty_quantification: false,
        ..Default::default()
    };

    let mut processor = NeuralBeamformingProcessor::new(config).unwrap();
    let rf_data = Array4::<f32>::ones((2, 8, 128, 1));

    let result = processor.process_volume(&rf_data);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.volume.dim(), (2, 8, 128));
    assert_eq!(output.uncertainty.dim(), (2, 8, 128));
    assert_eq!(output.confidence.dim(), (2, 8, 128));
}
