#[allow(unused_imports)]
use super::*;

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct TestBeamformingProvider;

#[cfg(feature = "gpu")]
impl super::super::provider::BeamformingGpuProvider for TestBeamformingProvider {
    fn provider_kind(&self) -> kwavers_solver::backend::traits::GpuProvider {
        kwavers_solver::backend::traits::GpuProvider::Wgpu
    }

    fn process_delay_and_sum(
        &self,
        config: &super::super::config::BeamformingConfig3D,
        rf_data: &leto::Array4<f32>,
        _dynamic_focusing: bool,
        apodization_window: &super::super::config::Beamforming3dApodizationWindow,
        _apodization_weights: &leto::Array3<f32>,
        _sub_volume_size: Option<(usize, usize, usize)>,
    ) -> kwavers_core::error::KwaversResult<leto::Array3<f32>> {
        super::super::delay_and_sum_cpu_reference(rf_data, config, apodization_window)
    }
}

#[test]
fn test_validate_input_empty_data() {
    let config = super::super::config::BeamformingConfig3D::default();
    #[cfg(feature = "gpu")]
    {
        use super::super::processor::BeamformingProcessor3D;
        let proc = BeamformingProcessor3D::with_provider(config, TestBeamformingProvider)
            .expect("test provider construction is infallible");
        let empty_data = leto::Array4::<f32>::zeros((0, 0, 0, 0));
        assert!(proc.validate_input(&empty_data).is_err());
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
        let proc = BeamformingProcessor3D::with_provider(config, TestBeamformingProvider)
            .expect("test provider construction is infallible");
        let bad_data = leto::Array4::<f32>::zeros((1, 100, 1024, 1));
        assert!(proc.validate_input(&bad_data).is_err());
    }

    #[cfg(not(feature = "gpu"))]
    {
        let expected_channels =
            config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2;
        assert_eq!(expected_channels, 16384);
    }
}
