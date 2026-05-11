use super::*;

#[test]
fn test_pca_fusion_uses_first_principal_component_for_correlated_modalities() {
    let config = FusionConfig {
        fusion_method: FusionMethod::PCA,
        uncertainty_quantification: true,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = (2, 2, 1);

    fusion.registered_data.insert(
        "a".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            quality_score: 0.6,
        },
    );
    fusion.registered_data.insert(
        "b".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![3.0, 4.0, 5.0, 6.0]).unwrap(),
            quality_score: 0.8,
        },
    );

    let fused = fusion.fuse().unwrap();
    let expected = Array3::from_shape_vec(shape, vec![2.0, 3.0, 4.0, 5.0]).unwrap();

    for (actual, expected) in fused.intensity_image.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
    for value in fused.confidence_map.iter() {
        assert!((value - 0.7).abs() < 1e-10);
    }
    for value in fused.uncertainty_map.unwrap().iter() {
        assert!((value - 0.3).abs() < 1e-10);
    }
}

#[test]
fn test_pca_fusion_selects_dominant_variance_modality() {
    let config = FusionConfig {
        fusion_method: FusionMethod::PCA,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 1, 1);

    fusion.registered_data.insert(
        "constant".to_string(),
        RegisteredModality {
            data: Array3::from_elem(shape, 10.0),
            quality_score: 0.1,
        },
    );
    fusion.registered_data.insert(
        "varying".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![0.0, 2.0, 4.0, 6.0]).unwrap(),
            quality_score: 0.9,
        },
    );

    let fused = fusion.fuse().unwrap();
    let expected = Array3::from_shape_vec(shape, vec![0.0, 2.0, 4.0, 6.0]).unwrap();

    for (actual, expected) in fused.intensity_image.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
    for value in fused.confidence_map.iter() {
        assert!((value - 0.9).abs() < 1e-10);
    }
}

#[test]
fn test_pca_fusion_rejects_nonfinite_data() {
    let config = FusionConfig {
        fusion_method: FusionMethod::PCA,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = (1, 2, 1);

    fusion.registered_data.insert(
        "finite".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![1.0, 2.0]).unwrap(),
            quality_score: 0.5,
        },
    );
    fusion.registered_data.insert(
        "invalid".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![1.0, f64::NAN]).unwrap(),
            quality_score: 0.5,
        },
    );

    let error = fusion.fuse().unwrap_err();

    assert!(format!("{error}").contains("finite intensity values"));
}
