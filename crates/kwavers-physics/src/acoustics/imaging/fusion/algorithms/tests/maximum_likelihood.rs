use super::*;

#[test]
fn test_maximum_likelihood_fusion_monotonicity() {
    let config = FusionConfig {
        fusion_method: ImagingFusionMethod::MaximumLikelihood,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);

    let shape = (5, 5, 2);

    // M-step should adaptively trust the cleaner signal
    fusion.registered_data.insert(
        "modality_clean".to_string(),
        RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.99, // Should have low variance initially
        },
    );

    fusion.registered_data.insert(
        "modality_noisy".to_string(),
        RegisteredModality {
            data: Array3::from_elem(shape, 10.0),
            quality_score: 0.01, // High variance
        },
    );

    let fused = fusion.fuse().unwrap();

    // Fused result should heavily favor the 'clean' signal
    for &v in fused.intensity_image.iter() {
        assert!(
            v < 3.0,
            "MLE should weight higher quality measurement higher"
        );
    }

    // Uncertainty map must be generated
    for &u in fused.uncertainty_map.as_ref().unwrap().iter() {
        assert!(u > 0.0, "Uncertainty must be strictly positive");
    }
}
