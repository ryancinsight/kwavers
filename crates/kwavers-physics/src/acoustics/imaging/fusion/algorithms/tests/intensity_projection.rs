use super::*;

#[test]
fn test_maximum_intensity_projection_selects_voxelwise_maximum() {
    let config = FusionConfig {
        fusion_method: ImagingFusionMethod::MaximumIntensity,
        uncertainty_quantification: true,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = [2, 2, 1];

    fusion.registered_data.insert(
        "a".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![1.0, 7.0, 3.0, 4.0]).unwrap(),
            quality_score: 0.25,
        },
    );
    fusion.registered_data.insert(
        "b".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![2.0, 5.0, 6.0, 1.0]).unwrap(),
            quality_score: 0.8,
        },
    );

    let fused = fusion.fuse().unwrap();

    assert_eq!(fused.intensity_image.shape(), shape);
    assert_eq!(
        fused.intensity_image.iter().copied().collect::<Vec<_>>(),
        vec![2.0_f64, 7.0, 6.0, 4.0]
    );
    assert_eq!(fused.confidence_map.shape(), shape);
    assert_eq!(
        fused.confidence_map.iter().copied().collect::<Vec<_>>(),
        vec![0.8_f64, 0.25, 0.8, 0.25]
    );
    let expected_uncertainty = Array3::from_shape_vec(shape, vec![0.2, 0.75, 0.2, 0.75]).unwrap();
    let uncertainty = fused.uncertainty_map.unwrap();
    for (actual, expected) in uncertainty.iter().zip(expected_uncertainty.iter()) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn test_minimum_intensity_projection_selects_voxelwise_minimum() {
    let config = FusionConfig {
        fusion_method: ImagingFusionMethod::MinimumIntensity,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = [2, 1, 1];

    fusion.registered_data.insert(
        "a".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![1.0, 7.0]).unwrap(),
            quality_score: 0.3,
        },
    );
    fusion.registered_data.insert(
        "b".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![2.0, 5.0]).unwrap(),
            quality_score: 0.9,
        },
    );

    let fused = fusion.fuse().unwrap();

    assert_eq!(fused.intensity_image.shape(), shape);
    assert_eq!(
        fused.intensity_image.iter().copied().collect::<Vec<_>>(),
        vec![1.0_f64, 5.0]
    );
    assert_eq!(fused.confidence_map.shape(), shape);
    assert_eq!(
        fused.confidence_map.iter().copied().collect::<Vec<_>>(),
        vec![0.3_f64, 0.9]
    );
    assert!(fused.uncertainty_map.is_none());
}

#[test]
fn test_intensity_projection_rejects_dimension_mismatch() {
    let config = FusionConfig {
        fusion_method: ImagingFusionMethod::MaximumIntensity,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);

    fusion.registered_data.insert(
        "a".to_string(),
        RegisteredModality {
            data: Array3::zeros([2, 2, 1]),
            quality_score: 0.5,
        },
    );
    fusion.registered_data.insert(
        "b".to_string(),
        RegisteredModality {
            data: Array3::zeros([2, 1, 1]),
            quality_score: 0.5,
        },
    );

    let error = fusion.fuse().unwrap_err();

    assert!(format!("{error}").contains("identical registered dimensions"));
}
