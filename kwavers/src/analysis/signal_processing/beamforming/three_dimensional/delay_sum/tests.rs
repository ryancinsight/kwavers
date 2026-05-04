#[cfg(not(feature = "gpu"))]
use super::processor::DelaySumGPU;
#[cfg(not(feature = "gpu"))]
use crate::analysis::signal_processing::beamforming::three_dimensional::config::BeamformingConfig3D;

#[test]
#[cfg(not(feature = "gpu"))]
fn test_element_positions_generation() {
    let config = BeamformingConfig3D::default();
    let delay_sum = DelaySumGPU::new(&config, (), (), (), ());
    let positions = delay_sum.create_element_positions();

    // Should have 3 floats per element (x, y, z)
    let expected_len =
        config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2 * 3;
    assert_eq!(positions.len(), expected_len);
}
