#[cfg(feature = "pinn")]
use crate::analysis::signal_processing::beamforming::neural::types::PINNBeamformingConfig;
#[cfg(feature = "pinn")]
use crate::solver::interface::pinn_beamforming::{
    DistributedConfig, LoadBalancingStrategy, PinnBeamformingDecompositionStrategy,
};

#[cfg(feature = "pinn")]
use super::processor::{DistributedNeuralBeamformingProcessor, FaultToleranceState};

#[cfg(feature = "pinn")]
#[test]
fn test_processor_creation() {
    let beamforming_config = PINNBeamformingConfig::default();
    let distributed_config = DistributedConfig {
        num_gpus: 1,
        gpu_devices: vec![0],
        batch_size_per_gpu: 32,
        decomposition: PinnBeamformingDecompositionStrategy::Spatial,
        load_balancing: LoadBalancingStrategy::Static,
    };
    let result = DistributedNeuralBeamformingProcessor::new(beamforming_config, distributed_config);

    let _ = result;
}

#[cfg(feature = "pinn")]
#[test]
fn test_fault_tolerance_default() {
    let ft = FaultToleranceState::default();
    assert_eq!(ft.max_retries, 3);
    assert!(ft.dynamic_load_balancing);
    assert_eq!(ft.load_imbalance_threshold, 0.2);
}

#[cfg(feature = "pinn")]
#[test]
fn test_fault_tolerance_config() {
    let mut fault_state = FaultToleranceState::default();
    assert_eq!(fault_state.max_retries, 3);
    assert!(fault_state.dynamic_load_balancing);

    fault_state.gpu_health = vec![true, true, true, true];
    fault_state.gpu_load = vec![0.5, 0.6, 0.4, 0.7];

    let max_load = fault_state.gpu_load.iter().copied().fold(0.0, f32::max);
    assert!(max_load <= 1.0);
}

#[cfg(feature = "pinn")]
#[test]
fn test_distributed_processing_matches_sequential_result() {
    use crate::analysis::signal_processing::beamforming::neural::pinn::NeuralBeamformingProcessor;
    use crate::analysis::signal_processing::beamforming::neural::types::PINNBeamformingConfig;
    use ndarray::Array4;

    let beamforming_config = PINNBeamformingConfig {
        rf_data_channels: 2,
        samples_per_channel: 3,
        volume_size: (4, 2, 3),
        enable_pinn: false,
        enable_uncertainty_quantification: false,
        ..Default::default()
    };

    let sequential_config = beamforming_config.clone();
    let mut sequential = NeuralBeamformingProcessor::new(sequential_config).unwrap();

    let distributed_config = DistributedConfig {
        num_gpus: 2,
        gpu_devices: vec![0, 1],
        batch_size_per_gpu: 1,
        decomposition: PinnBeamformingDecompositionStrategy::Spatial,
        load_balancing: LoadBalancingStrategy::Static,
    };
    let mut distributed =
        DistributedNeuralBeamformingProcessor::new(beamforming_config, distributed_config).unwrap();

    let rf_data = Array4::from_shape_fn((4, 2, 3, 1), |(frame, channel, sample, _)| {
        frame as f32 + 0.1 * channel as f32 + 0.01 * sample as f32
    });

    let expected = sequential.process_volume(&rf_data).unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let actual = runtime
        .block_on(distributed.process_volume_distributed(&rf_data))
        .unwrap();

    assert_eq!(actual.volume, expected.volume);
    assert_eq!(actual.uncertainty, expected.uncertainty);
    assert_eq!(actual.confidence, expected.confidence);
    assert_eq!(actual.num_gpus_used, 2);
    assert_eq!(distributed.metrics().active_gpus, 2);
    assert!(actual.load_balance_efficiency > 0.0);
    assert!(distributed.metrics().memory_efficiency > 0.0);
}
