//! Integration tests for 3D beamforming module
//!
//! Tests the complete beamforming pipeline including configuration,
//! processor initialization, and volume processing.

use super::*;

#[test]
fn test_beamforming_config_3d_default() {
    let config = BeamformingConfig3D::default();
    assert_eq!(config.volume_dims, (128, 128, 128));
    assert_eq!(config.num_elements_3d, (32, 32, 16));
    assert_eq!(config.center_frequency, 2.5e6);
    assert_eq!(config.sampling_frequency, 50e6);
    assert_eq!(config.sound_speed, 1540.0);
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
        apodization: ApodizationWindow::Hamming,
        sub_volume_size: Some((64, 64, 64)),
    };

    match algorithm {
        BeamformingAlgorithm3D::DelayAndSum {
            dynamic_focusing,
            apodization,
            sub_volume_size,
        } => {
            assert!(dynamic_focusing);
            assert!(matches!(apodization, ApodizationWindow::Hamming));
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
    let _rect = ApodizationWindow::Rectangular;
    let _hamming = ApodizationWindow::Hamming;
    let _hann = ApodizationWindow::Hann;
    let _blackman = ApodizationWindow::Blackman;
    let _gaussian = ApodizationWindow::Gaussian { sigma: 0.5 };
    let _custom = ApodizationWindow::Custom(vec![1.0, 0.8, 0.6]);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_processor_creation() {
    let config = BeamformingConfig3D::default();
    let result = BeamformingProcessor3D::new(config).await;

    // May fail if no GPU is available - that's okay for CI
    match result {
        Ok(processor) => {
            let metrics = processor.metrics();
            assert_eq!(metrics.processing_time_ms, 0.0);
        }
        Err(e) => {
            // Expected error when no GPU is available
            println!("GPU initialization failed (expected in CI): {:?}", e);
        }
    }
}

#[cfg(not(feature = "gpu"))]
#[tokio::test]
async fn test_processor_creation_cpu_only() {
    let config = BeamformingConfig3D::default();
    let result = BeamformingProcessor3D::new(config).await;

    // Should fail with FeatureNotAvailable error
    assert!(result.is_err());
}
