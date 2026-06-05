#[allow(unused_imports)]
use super::*;

#[test]
fn test_validate_input_empty_data() {
    let config = super::super::config::BeamformingConfig3D::default();
    #[cfg(feature = "gpu")]
    {
        use super::super::processor::BeamformingProcessor3D;
        let processor = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { BeamformingProcessor3D::new(config).await });
        if let Ok(proc) = processor {
            let empty_data = ndarray::Array4::<f32>::zeros((0, 0, 0, 0));
            assert!(proc.validate_input(&empty_data).is_err());
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        // CPU-only test - just verify config defaults
        assert_eq!(config.volume_dims, (128, 128, 128));
    }
}

#[test]
fn test_validate_input_channel_mismatch() {
    let config = super::super::config::BeamformingConfig3D::default();
    #[cfg(feature = "gpu")]
    {
        use super::super::processor::BeamformingProcessor3D;
        let processor = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { BeamformingProcessor3D::new(config).await });
        if let Ok(proc) = processor {
            let bad_data = ndarray::Array4::<f32>::zeros((1, 100, 1024, 1));
            assert!(proc.validate_input(&bad_data).is_err());
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        let expected_channels =
            config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2;
        assert_eq!(expected_channels, 16384);
    }
}
