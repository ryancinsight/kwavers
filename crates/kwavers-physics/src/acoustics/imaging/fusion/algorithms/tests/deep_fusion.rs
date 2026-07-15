use super::*;

#[test]
fn test_deep_fusion_attention_is_input_sensitive_and_convex() {
    let config = FusionConfig {
        fusion_method: ImagingFusionMethod::DeepFusion,
        uncertainty_quantification: true,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = [2, 1, 1];

    fusion.registered_data.insert(
        "a".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![0.0, 1.0]).unwrap(),
            quality_score: 0.5,
        },
    );
    fusion.registered_data.insert(
        "b".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![1.0, 0.0]).unwrap(),
            quality_score: 0.5,
        },
    );

    let fused = fusion.fuse().unwrap();
    let expected_high = std::f64::consts::E / (1.0 + std::f64::consts::E);
    let expected_low = 1.0 / (1.0 + std::f64::consts::E);

    assert!((fused.intensity_image[[0, 0, 0]] - expected_high).abs() < 1e-12);
    assert!((fused.intensity_image[[1, 0, 0]] - expected_high).abs() < 1e-12);
    assert!(fused
        .intensity_image
        .iter()
        .all(|value| *value >= 0.0 && *value <= 1.0));
    assert!(fused
        .confidence_map
        .iter()
        .all(|value| (*value - 0.5).abs() < 1e-12));

    let uncertainty = fused.uncertainty_map.unwrap();
    let expected_entropy =
        -(expected_high * expected_high.ln() + expected_low * expected_low.ln()) / 2.0_f64.ln();
    assert!(uncertainty
        .iter()
        .all(|value| (*value - expected_entropy).abs() < 1e-12));
}

#[test]
fn test_deep_fusion_quality_prior_changes_attention_weights() {
    let config = FusionConfig {
        fusion_method: ImagingFusionMethod::DeepFusion,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = [2, 1, 1];

    fusion.registered_data.insert(
        "low_quality".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![0.0, 10.0]).unwrap(),
            quality_score: 0.1,
        },
    );
    fusion.registered_data.insert(
        "high_quality".to_string(),
        RegisteredModality {
            data: Array3::from_shape_vec(shape, vec![2.0, 2.0]).unwrap(),
            quality_score: 0.9,
        },
    );

    let fused = fusion.fuse().unwrap();

    assert!(fused.intensity_image[[0, 0, 0]] > 1.7);
    assert!(fused.intensity_image[[1, 0, 0]] < 4.5);
    assert!(fused.confidence_map[[0, 0, 0]] > fused.confidence_map[[1, 0, 0]]);
}

#[test]
fn test_deep_fusion_rejects_nonfinite_data() {
    let config = FusionConfig {
        fusion_method: ImagingFusionMethod::DeepFusion,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = [1, 2, 1];

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
            data: Array3::from_shape_vec(shape, vec![1.0, f64::INFINITY]).unwrap(),
            quality_score: 0.5,
        },
    );

    let error = fusion.fuse().unwrap_err();

    assert!(format!("{error}").contains("finite intensity values"));
}
