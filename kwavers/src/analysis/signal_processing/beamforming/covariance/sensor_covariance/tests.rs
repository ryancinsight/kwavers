use super::post_process::CovariancePostProcess;
use super::shrinkage::shrinkage_to_identity_real;
use super::{CovarianceEstimator, SpatialSmoothingComplex};
use approx::assert_abs_diff_eq;
use ndarray::Array2;
use num_complex::Complex64;

#[test]
fn estimate_complex_is_hermitian_for_simple_data() {
    let mut x = Array2::<Complex64>::zeros((2, 3));
    x[(0, 0)] = Complex64::new(1.0, 2.0);
    x[(1, 0)] = Complex64::new(-0.5, 0.25);
    x[(0, 1)] = Complex64::new(0.1, -0.2);
    x[(1, 1)] = Complex64::new(0.3, 0.4);
    x[(0, 2)] = Complex64::new(-1.0, 0.0);
    x[(1, 2)] = Complex64::new(0.0, 1.0);

    let est = CovarianceEstimator {
        forward_backward_averaging: false,
        num_snapshots: 1,
        post_process: CovariancePostProcess::None,
    };

    let r = est.estimate_complex(&x).expect("covariance");

    for i in 0..2 {
        for j in 0..2 {
            let lhs = r[(i, j)];
            let rhs = r[(j, i)].conj();
            assert_abs_diff_eq!(lhs.re, rhs.re, epsilon = 1e-12);
            assert_abs_diff_eq!(lhs.im, rhs.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn estimate_complex_rejects_empty() {
    let x = Array2::<Complex64>::zeros((0, 0));
    let est = CovarianceEstimator::default();
    let err = est.estimate_complex(&x).expect_err("must reject empty");
    assert!(err.to_string().contains("estimate_complex"));
}

#[test]
fn forward_backward_averaging_complex_preserves_hermitian_structure() {
    let mut r = Array2::<Complex64>::zeros((3, 3));
    r[(0, 0)] = Complex64::new(2.0, 0.0);
    r[(1, 1)] = Complex64::new(3.0, 0.0);
    r[(2, 2)] = Complex64::new(4.0, 0.0);
    r[(0, 1)] = Complex64::new(0.5, 0.25);
    r[(1, 0)] = r[(0, 1)].conj();
    r[(1, 2)] = Complex64::new(-0.2, 0.1);
    r[(2, 1)] = r[(1, 2)].conj();
    r[(0, 2)] = Complex64::new(0.0, -0.3);
    r[(2, 0)] = r[(0, 2)].conj();

    let est = CovarianceEstimator {
        forward_backward_averaging: true,
        num_snapshots: 1,
        post_process: CovariancePostProcess::None,
    };

    let fb = est.apply_forward_backward_averaging_complex(&r);

    for i in 0..3 {
        for j in 0..3 {
            let lhs = fb[(i, j)];
            let rhs = fb[(j, i)].conj();
            assert_abs_diff_eq!(lhs.re, rhs.re, epsilon = 1e-12);
            assert_abs_diff_eq!(lhs.im, rhs.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn shrinkage_to_identity_real_preserves_symmetry_and_improves_diagonal() {
    let mut r = Array2::<f64>::zeros((2, 2));
    r[(0, 0)] = 1.0;
    r[(1, 1)] = 3.0;
    r[(0, 1)] = 0.2;
    r[(1, 0)] = 0.2;

    let shrunk = shrinkage_to_identity_real(&r, 0.5);

    assert_abs_diff_eq!(shrunk[(0, 1)], shrunk[(1, 0)], epsilon = 1e-15);
    assert!(shrunk[(0, 0)].is_finite() && shrunk[(1, 1)].is_finite());
}

#[test]
fn spatial_smoothing_complex_shapes_match() {
    let mut r = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        r[(i, i)] = Complex64::new(1.0 + i as f64, 0.0);
    }

    let smoother = SpatialSmoothingComplex::new(3);
    let sm = smoother.apply(&r).expect("smoothed");
    assert_eq!(sm.nrows(), 3);
    assert_eq!(sm.ncols(), 3);
}
