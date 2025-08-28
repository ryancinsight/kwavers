//! Tests for AMR module

use super::*;
use ndarray::Array3;

#[test]
fn test_octree_creation() {
    let bounds = crate::grid::Bounds::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let octree = Octree::new(bounds, 3).unwrap();

    assert_eq!(octree.node_count(), 1);
    assert_eq!(octree.leaf_count(), 1);
}

#[test]
fn test_refinement_manager() {
    let manager = RefinementManager::new(4);
    let error = Array3::ones((8, 8, 8));

    let markers = manager.mark_cells(&error, 0.5).unwrap();
    assert_eq!(markers.dim(), (8, 8, 8));
}

#[test]
fn test_interpolation() {
    let interpolator = ConservativeInterpolator::new();
    let coarse = Array3::ones((4, 4, 4));

    let fine = interpolator.prolongate(&coarse);
    assert_eq!(fine.dim(), (8, 8, 8));

    let restricted = interpolator.restrict(&fine);
    assert_eq!(restricted.dim(), (4, 4, 4));
}

#[test]
fn test_error_estimation() {
    let estimator = ErrorEstimator::new();
    let field = Array3::zeros((8, 8, 8));

    let error = estimator.estimate_error(&field).unwrap();
    assert_eq!(error.dim(), field.dim());
}

#[test]
fn test_wavelet_transform() {
    let wavelet = WaveletTransform::new(WaveletBasis::Haar, 2);
    let data = Array3::ones((8, 8, 8));

    let coeffs = wavelet.forward(&data).unwrap();
    assert_eq!(coeffs.dim(), data.dim());

    let reconstructed = wavelet.inverse(&coeffs).unwrap();
    assert_eq!(reconstructed.dim(), data.dim());
}
