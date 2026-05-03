//! Custom weights, probabilistic, and non-rigid fusion tests.

use super::super::*;
use ndarray::Array3;

#[test]
fn test_fusion_with_custom_weights() {
    let config = FusionConfig {
        modality_weights: [
            ("ultrasound".to_string(), 0.7),
            ("photoacoustic".to_string(), 0.3),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };

    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 10.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse().unwrap();

    // Expected: (0.7 * 1.0 + 0.3 * 10.0) / 1.0 = 3.7
    let expected = 3.7;
    for value in result.intensity_image.iter() {
        assert!((value - expected).abs() < 1e-9);
    }
}

#[test]
fn test_probabilistic_fusion_uncertainty() {
    let config = FusionConfig {
        fusion_method: FusionMethod::Probabilistic,
        ..Default::default()
    };

    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 3.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse().unwrap();

    // Probabilistic fusion should always provide uncertainty
    assert!(result.uncertainty_map.is_some());

    let uncertainty = result.uncertainty_map.unwrap();
    assert_eq!(uncertainty.dim(), shape);

    // Should have non-zero uncertainty due to variance between modalities
    for value in uncertainty.iter() {
        assert!(*value > 0.0);
        assert!(*value <= 1.0);
    }
}

#[test]
fn test_nonrigid_fusion_succeeds() {
    // NonRigid is implemented via Symmetric Gaussian Demons (Vercauteren 2009).
    let config = FusionConfig {
        registration_method: RegistrationMethod::NonRigid,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse();
    assert!(result.is_ok(), "NonRigid fusion should succeed: {result:?}");
}
