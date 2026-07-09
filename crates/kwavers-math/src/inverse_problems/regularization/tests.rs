//! Tests for regularization strategies.

use super::*;

#[test]
fn test_regularization_config_default() {
    let cfg = RegularizationConfig::default();
    assert_eq!(cfg.tikhonov_weight, 0.01);
    assert_eq!(cfg.tv_weight, 0.0);
    assert!(cfg.is_active());
}

#[test]
fn test_regularization_config_builder() {
    let cfg = RegularizationConfig::none()
        .with_tikhonov(0.05)
        .with_tv(0.02)
        .with_smoothness(0.01);

    assert_eq!(cfg.tikhonov_weight, 0.05);
    assert_eq!(cfg.tv_weight, 0.02);
    assert_eq!(cfg.smoothness_weight, 0.01);
    assert!(cfg.is_active());
}

#[test]
fn test_tikhonov_3d() {
    use leto::Array3;
    let mut gradient = Array3::zeros((3, 3, 3));
    let model = Array3::ones((3, 3, 3));

    let cfg = RegularizationConfig::default().with_tikhonov(0.5);
    let regularizer = ModelRegularizer3D::new(cfg);
    regularizer.apply_to_gradient(&mut gradient, &model);

    assert!(gradient[[1, 1, 1]] > 0.0);
}

#[test]
fn test_tv_3d_basic() {
    use leto::Array3;
    let mut gradient = Array3::zeros((3, 3, 3));
    let model = Array3::zeros((3, 3, 3));

    let cfg = RegularizationConfig::default().with_tv(0.1);
    let regularizer = ModelRegularizer3D::new(cfg);
    regularizer.apply_to_gradient(&mut gradient, &model);

    assert_eq!(gradient[[1, 1, 1]], 0.0);
}

#[test]
fn test_smoothness_3d() {
    use leto::Array3;
    let mut gradient = Array3::zeros((5, 5, 5));
    gradient[[2, 2, 2]] = 1.0;

    let cfg = RegularizationConfig::default().with_smoothness(0.1);
    let regularizer = ModelRegularizer3D::new(cfg);
    let model = Array3::zeros((5, 5, 5));
    regularizer.apply_to_gradient(&mut gradient, &model);

    assert!(gradient[[2, 2, 2]] < 1.0);
}

#[test]
fn test_l1_3d() {
    use leto::Array3;
    let mut gradient = Array3::zeros((3, 3, 3));
    let model = Array3::ones((3, 3, 3));

    let cfg = RegularizationConfig::default().with_l1(0.5);
    let regularizer = ModelRegularizer3D::new(cfg);
    regularizer.apply_to_gradient(&mut gradient, &model);

    assert!(gradient[[1, 1, 1]] > 0.0);
}

#[test]
fn test_regularization_2d() {
    use leto::Array2;
    let mut gradient = Array2::zeros((3, 3));
    let model = Array2::ones((3, 3));

    let cfg = RegularizationConfig::default().with_tikhonov(0.2);
    let regularizer = ModelRegularizer2D::new(cfg);
    regularizer.apply_to_gradient(&mut gradient, &model);

    assert!(gradient[[1, 1]] > 0.0);
}

#[test]
fn test_regularization_1d() {
    use leto::Array1;
    let mut gradient = Array1::zeros(5);
    let model = Array1::ones(5);

    let cfg = RegularizationConfig::default().with_tikhonov(0.3);
    let regularizer = ModelRegularizer1D::new(cfg);
    regularizer.apply_to_gradient(&mut gradient, &model);

    assert!(gradient[2] > 0.0);
}

#[test]
fn test_combined_regularization() {
    use leto::Array3;
    let mut gradient = Array3::zeros((5, 5, 5));
    let model = Array3::ones((5, 5, 5));

    let cfg = RegularizationConfig::default()
        .with_tikhonov(0.01)
        .with_tv(0.01)
        .with_smoothness(0.01);

    let regularizer = ModelRegularizer3D::new(cfg);
    regularizer.apply_to_gradient(&mut gradient, &model);

    assert!(gradient[[2, 2, 2]] > 0.0);
}

#[test]
fn test_no_regularization() {
    use leto::Array3;
    let mut gradient = Array3::ones((3, 3, 3));
    let model = Array3::ones((3, 3, 3));
    let original = gradient.clone();

    let cfg = RegularizationConfig::none();
    let regularizer = ModelRegularizer3D::new(cfg);
    regularizer.apply_to_gradient(&mut gradient, &model);

    assert_eq!(gradient, original);
}
