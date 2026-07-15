use super::*;

#[test]
fn test_register_optical_validation() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    // Negative invariant: Optical intensity cannot be negative
    let mut invalid_data = Array3::<f64>::zeros([4, 4, 2]);
    invalid_data[[0, 0, 0]] = -1.0;

    let result = fusion.register_optical(&invalid_data, 550e-9);
    assert!(
        result.is_err(),
        "Must enforce physical constraint: Intensity >= 0"
    );

    let valid_data = Array3::<f64>::ones([4, 4, 2]);
    fusion.register_optical(&valid_data, 550e-9).unwrap();
}

#[test]
fn test_robust_bounds_degenerate() {
    let empty_data = Array3::<f64>::zeros([0, 0, 0]);
    let (min, max) = utils::compute_robust_bounds(&empty_data);
    assert_eq!(min, 0.0);
    assert_eq!(max, 0.0);

    let uniform_data = Array3::<f64>::ones([2, 2, 2]);
    let (min, max) = utils::compute_robust_bounds(&uniform_data);
    assert_eq!(min, 1.0);
    assert_eq!(max, 2.0);
}
