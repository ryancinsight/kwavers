use super::layer::BoundaryLayer;
use super::types::BoundaryDirections;

#[test]
fn test_boundary_directions() {
    let all = BoundaryDirections::all();
    assert!(all.x_min && all.x_max);

    let none = BoundaryDirections::none();
    assert!(!none.x_min && !none.x_max);

    let xy = BoundaryDirections::xy_plane();
    assert!(xy.x_min && !xy.z_min);
}

#[test]
fn test_boundary_layer() {
    let layer = BoundaryLayer::new(0, 10, 0, false);
    assert_eq!(layer.thickness, 10);
    assert!(layer.contains(5));
    assert!(!layer.contains(15));

    assert!((layer.normalized_distance(0) - 0.0).abs() < 1e-10);
    assert!((layer.normalized_distance(9) - 1.0).abs() < 1e-10);
}

#[test]
fn test_polynomial_profile() {
    let layer = BoundaryLayer::new(0, 10, 0, false);
    let sigma_max = 100.0;

    let edge_sigma = layer.polynomial_profile(0, 2, sigma_max);
    assert!((edge_sigma - sigma_max).abs() < 1e-10);

    let interior_sigma = layer.polynomial_profile(9, 2, sigma_max);
    assert!(interior_sigma.abs() < 1e-10);

    let mid_sigma = layer.polynomial_profile(5, 2, sigma_max);
    assert!(mid_sigma > interior_sigma && mid_sigma < edge_sigma);
}
