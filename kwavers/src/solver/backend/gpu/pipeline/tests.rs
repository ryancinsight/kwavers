use super::types::PipelineType;

#[test]
fn test_pipeline_type_enum() {
    assert_ne!(PipelineType::FFT3D, PipelineType::IFFT3D);
    assert_ne!(PipelineType::FFT3D, PipelineType::ElementWiseMultiply);
}
