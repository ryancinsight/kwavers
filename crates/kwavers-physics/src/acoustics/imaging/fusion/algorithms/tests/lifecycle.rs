use super::*;

#[test]
fn test_multimodal_fusion_creation() {
    let config = FusionConfig::default();
    let fusion = MultiModalFusion::new(config);
    assert_eq!(fusion.num_registered_modalities(), 0);
    assert!(fusion.registered_data.is_empty());
}

#[test]
fn test_register_ultrasound() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    let data = Array3::<f64>::zeros((8, 8, 4));
    fusion.register_ultrasound(&data).unwrap();
    assert_eq!(fusion.num_registered_modalities(), 1);
    assert!(fusion.is_modality_registered("ultrasound"));
}

#[test]
fn test_fuse_insufficient_modalities() {
    let config = FusionConfig::default();
    let fusion = MultiModalFusion::new(config);
    let result = fusion.fuse();

    assert!(result.is_err());
    if let Err(KwaversError::Validation(e)) = result {
        assert!(e.to_string().contains("At least two modalities"));
    } else {
        panic!("Expected Validation ConstraintViolation error.");
    }
}
