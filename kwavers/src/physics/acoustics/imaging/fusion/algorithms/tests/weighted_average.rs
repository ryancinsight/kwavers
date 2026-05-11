use super::*;

#[test]
fn test_weighted_average_fusion_is_bounded() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    let shape = (4, 4, 4);

    // Bounds check property test: Output must strictly be within [min(A_i), max(B_i)]
    let v_us = 1.0;
    let v_pa = 5.0;

    fusion
        .register_ultrasound(&Array3::from_elem(shape, v_us))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        RegisteredModality {
            data: Array3::from_elem(shape, v_pa),
            quality_score: 0.8,
        },
    );

    let fused = fusion.fuse().unwrap();
    assert_eq!(fused.intensity_image.dim(), shape);

    let weights = &fusion.config.modality_weights;
    let w_us = weights.get("ultrasound").copied().unwrap_or(1.0);
    let w_pa = weights.get("photoacoustic").copied().unwrap_or(1.0);
    let expected = (w_us * v_us + w_pa * v_pa) / (w_us + w_pa);

    for &v in fused.intensity_image.iter() {
        assert!((v - expected).abs() < 1e-9);
        assert!(v >= v_us && v <= v_pa);
    }
}
