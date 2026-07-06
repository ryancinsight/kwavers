use super::types::PipelineType;

#[test]
fn test_pipeline_type_enum() {
    // FFT3D/IFFT3D are owned by Apollo's FFT backend contract;
    // the elementwise/operators pipeline now exposes these two variants.
    assert_ne!(
        PipelineType::ElementWiseMultiply,
        PipelineType::SpatialDerivative
    );
}
