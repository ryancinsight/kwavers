use super::types::PipelineType;

#[test]
fn test_pipeline_type_enum() {
    // FFT3D/IFFT3D were removed when FFT moved to the `apollofft-wgpu` backend;
    // the elementwise/operators pipeline now exposes these two variants.
    assert_ne!(
        PipelineType::ElementWiseMultiply,
        PipelineType::SpatialDerivative
    );
}
