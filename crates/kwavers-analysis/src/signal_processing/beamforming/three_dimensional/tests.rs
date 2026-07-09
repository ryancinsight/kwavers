//! Integration tests for 3D beamforming module
//!
//! Tests the complete beamforming pipeline including configuration,
//! processor initialization, and volume processing.

use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

use super::*;

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct TestBeamformingProvider;

#[cfg(feature = "gpu")]
impl BeamformingGpuProvider for TestBeamformingProvider {
    fn provider_kind(&self) -> kwavers_solver::backend::traits::GpuProvider {
        kwavers_solver::backend::traits::GpuProvider::Wgpu
    }

    fn process_delay_and_sum(
        &self,
        config: &BeamformingConfig3D,
        rf_data: &leto::Array4<f32>,
        _dynamic_focusing: bool,
        apodization_window: &Beamforming3dApodizationWindow,
        _apodization_weights: &leto::Array3<f32>,
        _sub_volume_size: Option<(usize, usize, usize)>,
    ) -> kwavers_core::error::KwaversResult<leto::Array3<f32>> {
        delay_and_sum_cpu_reference(rf_data, config, apodization_window)
    }
}

#[test]
fn test_beamforming_config_3d_default() {
    let config = BeamformingConfig3D::default();
    assert_eq!(config.volume_dims, (128, 128, 128));
    assert_eq!(config.num_elements_3d, (32, 32, 16));
    assert_eq!(config.center_frequency, 2.5 * MHZ_TO_HZ);
    assert_eq!(config.sampling_frequency, 50e6);
    assert_eq!(config.sound_speed, SOUND_SPEED_TISSUE);
}

#[test]
fn test_beamforming_metrics_default() {
    let metrics = BeamformingMetrics::default();
    assert_eq!(metrics.processing_time_ms, 0.0);
    assert_eq!(metrics.gpu_memory_mb, 0.0);
    assert_eq!(metrics.cpu_memory_mb, 0.0);
    assert_eq!(metrics.reconstruction_rate, 0.0);
    assert_eq!(metrics.dynamic_range_db, 0.0);
    assert_eq!(metrics.snr_db, 0.0);
}

#[test]
fn test_algorithm_delay_and_sum() {
    let algorithm = BeamformingAlgorithm3D::DelayAndSum {
        dynamic_focusing: true,
        apodization: Beamforming3dApodizationWindow::Hamming,
        sub_volume_size: Some((64, 64, 64)),
    };

    match algorithm {
        BeamformingAlgorithm3D::DelayAndSum {
            dynamic_focusing,
            apodization,
            sub_volume_size,
        } => {
            assert!(dynamic_focusing);
            assert!(matches!(
                apodization,
                Beamforming3dApodizationWindow::Hamming
            ));
            assert_eq!(sub_volume_size, Some((64, 64, 64)));
        }
        _ => panic!("Expected DelayAndSum algorithm"),
    }
}

#[test]
fn test_algorithm_mvdr_3d() {
    let algorithm = BeamformingAlgorithm3D::MVDR3D {
        diagonal_loading: 0.01,
        subarray_size: [8, 8, 4],
    };

    match algorithm {
        BeamformingAlgorithm3D::MVDR3D {
            diagonal_loading,
            subarray_size,
        } => {
            assert_eq!(diagonal_loading, 0.01);
            assert_eq!(subarray_size, [8, 8, 4]);
        }
        _ => panic!("Expected MVDR3D algorithm"),
    }
}

#[test]
fn test_apodization_window_types() {
    let _rect = Beamforming3dApodizationWindow::Rectangular;
    let _hamming = Beamforming3dApodizationWindow::Hamming;
    let _hann = Beamforming3dApodizationWindow::Hann;
    let _blackman = Beamforming3dApodizationWindow::Blackman;
    let _gaussian = Beamforming3dApodizationWindow::Gaussian { sigma: 0.5 };
    let _custom = Beamforming3dApodizationWindow::Custom(vec![1.0, 0.8, 0.6]);
}

#[cfg(feature = "gpu")]
#[test]
fn test_processor_creation() {
    let config = BeamformingConfig3D::default();
    let processor = BeamformingProcessor3D::with_provider(config, TestBeamformingProvider)
        .expect("test provider construction is infallible");
    let metrics = processor.metrics();
    assert_eq!(metrics.processing_time_ms, 0.0);
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_processor_creation_cpu_only() {
    let config = BeamformingConfig3D::default();
    let result = BeamformingProcessor3D::new_wgpu(config);

    // Should fail with FeatureNotAvailable error
    assert!(result.is_err());
}

/// Contract validation: provider-injected static DAS must agree with the
/// analytically-specified CPU reference. The real WGPU differential test lives
/// in `kwavers-gpu`, where the concrete provider is linked.
#[cfg(feature = "gpu")]
#[test]
fn das_provider_contract_matches_cpu_reference() {
    use leto::Array4;

    let config = BeamformingConfig3D {
        num_elements_3d: (2, 1, 2),
        element_spacing_3d: (1.0e-3, 1.0e-3, 1.0e-3),
        volume_dims: (4, 1, 4),
        voxel_spacing: (1.0e-3, 1.0e-3, 1.0e-3),
        sound_speed: 1500.0,
        sampling_frequency: 10.0e6,
        ..BeamformingConfig3D::default()
    };

    // Deterministic RF: 1 frame, 4 channels (= 2·1·2), 64 samples.
    let mut rf = Array4::<f32>::zeros((1, 4, 64, 1));
    for ch in 0..4 {
        for s in 0..64 {
            rf[[0, ch, s, 0]] = (((ch * 13 + s * 7) % 17) as f32) - 8.0;
        }
    }

    let cpu =
        delay_and_sum_cpu_reference(&rf, &config, &Beamforming3dApodizationWindow::Rectangular)
            .expect("CPU DAS reference");

    let mut processor =
        BeamformingProcessor3D::with_provider(config.clone(), TestBeamformingProvider)
            .expect("test provider construction is infallible");

    let gpu = processor
        .process_volume(
            &rf,
            &BeamformingAlgorithm3D::DelayAndSum {
                dynamic_focusing: false,
                apodization: Beamforming3dApodizationWindow::Rectangular,
                sub_volume_size: None,
            },
        )
        .expect("GPU DAS reconstruction");

    assert_eq!(cpu.dim(), gpu.dim(), "CPU/GPU volume dimensions differ");
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        let tol = 1.0e-3f32.mul_add(c.abs(), 1.0e-2);
        assert!(
            (c - g).abs() <= tol,
            "provider DAS diverges from CPU reference: cpu={c}, provider={g}, tol={tol}"
        );
    }
}
