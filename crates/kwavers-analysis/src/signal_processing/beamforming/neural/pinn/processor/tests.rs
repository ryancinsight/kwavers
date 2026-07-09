use super::core::NeuralBeamformingProcessor;
use crate::signal_processing::beamforming::neural::types::PINNBeamformingConfig;

#[test]
fn test_processor_creation() {
    // default: rf_data_channels=64, samples_per_channel=1024, volume_size=(1,64,1024)
    let config = PINNBeamformingConfig::default();
    let processor = NeuralBeamformingProcessor::new(config).unwrap();
    assert_eq!(
        processor.config.rf_data_channels, 64,
        "default rf_data_channels must be 64"
    );
    assert_eq!(
        processor.config.samples_per_channel, 1024,
        "default samples_per_channel must be 1024"
    );
    // Fresh processor: empty steering cache, zero metrics
    assert_eq!(
        processor.steering_cache_len(),
        0,
        "new processor must have empty steering cache"
    );
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
    use leto::Array4;

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

    let output = processor.process_volume(&rf_data).unwrap();
    assert_eq!(output.volume.dim(), (2, 8, 128));
    assert_eq!(output.uncertainty.dim(), (2, 8, 128));
    assert_eq!(output.confidence.dim(), (2, 8, 128));
}
