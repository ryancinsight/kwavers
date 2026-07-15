use super::post_process::CovariancePostProcess;
use super::shrinkage::shrinkage_to_identity_real;
use super::{CovarianceEstimator, SpatialSmoothingComplex};
use approx::assert_abs_diff_eq;
use eunomia::Complex64;
use leto::Array2;

#[test]
fn estimate_complex_is_hermitian_for_simple_data() {
    let mut x = Array2::<Complex64>::from_elem((2, 3), Complex64::default());
    x[[0, 0]] = Complex64::new(1.0, 2.0);
    x[[1, 0]] = Complex64::new(-0.5, 0.25);
    x[[0, 1]] = Complex64::new(0.1, -0.2);
    x[[1, 1]] = Complex64::new(0.3, 0.4);
    x[[0, 2]] = Complex64::new(-1.0, 0.0);
    x[[1, 2]] = Complex64::new(0.0, 1.0);

    let est = CovarianceEstimator {
        forward_backward_averaging: false,
        num_snapshots: 1,
        post_process: CovariancePostProcess::None,
    };

    let r = est.estimate_complex(&x).expect("covariance");

    for i in 0..2 {
        for j in 0..2 {
            let lhs = r[[i, j]];
            let rhs = r[[j, i]].conj();
            assert_abs_diff_eq!(lhs.re, rhs.re, epsilon = 1e-12);
            assert_abs_diff_eq!(lhs.im, rhs.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn estimate_complex_rejects_empty() {
    let x = Array2::<Complex64>::from_elem((0, 0), Complex64::default());
    let est = CovarianceEstimator::default();
    let err = est.estimate_complex(&x).expect_err("must reject empty");
    assert!(err.to_string().contains("estimate_complex"));
}

#[test]
fn forward_backward_averaging_complex_preserves_hermitian_structure() {
    let mut r = Array2::<Complex64>::from_elem((3, 3), Complex64::default());
    r[[0, 0]] = Complex64::new(2.0, 0.0);
    r[[1, 1]] = Complex64::new(3.0, 0.0);
    r[[2, 2]] = Complex64::new(4.0, 0.0);
    r[[0, 1]] = Complex64::new(0.5, 0.25);
    r[[1, 0]] = r[[0, 1]].conj();
    r[[1, 2]] = Complex64::new(-0.2, 0.1);
    r[[2, 1]] = r[[1, 2]].conj();
    r[[0, 2]] = Complex64::new(0.0, -0.3);
    r[[2, 0]] = r[[0, 2]].conj();

    let est = CovarianceEstimator {
        forward_backward_averaging: true,
        num_snapshots: 1,
        post_process: CovariancePostProcess::None,
    };

    let fb = est.apply_forward_backward_averaging_complex(&r);

    for i in 0..3 {
        for j in 0..3 {
            let lhs = fb[[i, j]];
            let rhs = fb[[j, i]].conj();
            assert_abs_diff_eq!(lhs.re, rhs.re, epsilon = 1e-12);
            assert_abs_diff_eq!(lhs.im, rhs.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn shrinkage_to_identity_real_preserves_symmetry_and_improves_diagonal() {
    let mut r = Array2::<f64>::zeros((2, 2));
    r[[0, 0]] = 1.0;
    r[[1, 1]] = 3.0;
    r[[0, 1]] = 0.2;
    r[[1, 0]] = 0.2;

    let shrunk = shrinkage_to_identity_real(&r, 0.5);

    assert_abs_diff_eq!(shrunk[[0, 1]], shrunk[[1, 0]], epsilon = 1e-15);
    assert!(shrunk[[0, 0]].is_finite() && shrunk[[1, 1]].is_finite());
}

// ─── Shrinkage: exact closed-form verification ────────────────────────────────

/// Shrinkage of a 2×2 diagonal-dominant matrix at alpha=0.5.
///
/// Inputs: R = [[1, 0.2], [0.2, 3]], alpha = 0.5.
/// trace = 4, mu = trace/2 = 2.
/// out[i,j] = (1-0.5)·R[i,j]         for i≠j  → 0.1
/// out[i,i] = (1-0.5)·R[i,i] + 0.5·2 → 1.5, 2.5
#[test]
fn shrinkage_to_identity_real_exact_values() {
    let mut r = Array2::<f64>::zeros((2, 2));
    r[[0, 0]] = 1.0;
    r[[1, 1]] = 3.0;
    r[[0, 1]] = 0.2;
    r[[1, 0]] = 0.2;

    let shrunk = shrinkage_to_identity_real(&r, 0.5);

    assert_abs_diff_eq!(shrunk[[0, 0]], 1.5, epsilon = 1e-14);
    assert_abs_diff_eq!(shrunk[[1, 1]], 2.5, epsilon = 1e-14);
    assert_abs_diff_eq!(shrunk[[0, 1]], 0.1, epsilon = 1e-14);
    assert_abs_diff_eq!(shrunk[[1, 0]], 0.1, epsilon = 1e-14);
}

/// Full-strength shrinkage (alpha=1) maps any matrix to mu·I where mu = trace/m.
///
/// For R = [[1, 0.2], [0.2, 3]]:
///   trace = 4, mu = 2.
///   Result = 0·R + 1·mu·I = [[2, 0], [0, 2]].
#[test]
fn shrinkage_to_identity_real_full_strength_gives_scaled_identity() {
    let mut r = Array2::<f64>::zeros((2, 2));
    r[[0, 0]] = 1.0;
    r[[1, 1]] = 3.0;
    r[[0, 1]] = 0.2;
    r[[1, 0]] = 0.2;

    let shrunk = shrinkage_to_identity_real(&r, 1.0);

    assert_abs_diff_eq!(shrunk[[0, 0]], 2.0, epsilon = 1e-14);
    assert_abs_diff_eq!(shrunk[[1, 1]], 2.0, epsilon = 1e-14);
    assert_abs_diff_eq!(shrunk[[0, 1]], 0.0, epsilon = 1e-14);
    assert_abs_diff_eq!(shrunk[[1, 0]], 0.0, epsilon = 1e-14);
}

/// Zero shrinkage (alpha=0) returns the original matrix unchanged.
#[test]
fn shrinkage_to_identity_real_zero_alpha_is_identity_transform() {
    let mut r = Array2::<f64>::zeros((2, 2));
    r[[0, 0]] = 5.0;
    r[[1, 1]] = 7.0;
    r[[0, 1]] = 1.3;
    r[[1, 0]] = 1.3;

    let shrunk = shrinkage_to_identity_real(&r, 0.0);

    assert_abs_diff_eq!(shrunk[[0, 0]], 5.0, epsilon = 1e-14);
    assert_abs_diff_eq!(shrunk[[1, 1]], 7.0, epsilon = 1e-14);
    assert_abs_diff_eq!(shrunk[[0, 1]], 1.3, epsilon = 1e-14);
}

// ─── CovarianceEstimator: exact outer-product verification ───────────────────

/// Single snapshot [1, 2]ᵀ → R = x·xᵀ / 1 = [[1, 2], [2, 4]].
///
/// With one snapshot, N=1, the sample covariance is exactly the outer product.
/// No forward-backward averaging, no post-processing.
#[test]
fn estimate_single_snapshot_gives_exact_outer_product() {
    let mut x = Array2::<f64>::zeros((2, 1));
    x[[0, 0]] = 1.0;
    x[[1, 0]] = 2.0;

    let est = CovarianceEstimator {
        forward_backward_averaging: false,
        num_snapshots: 1,
        post_process: CovariancePostProcess::None,
    };
    let r = est.estimate(&x).unwrap();

    assert_abs_diff_eq!(r[[0, 0]], 1.0, epsilon = 1e-14);
    assert_abs_diff_eq!(r[[0, 1]], 2.0, epsilon = 1e-14);
    assert_abs_diff_eq!(r[[1, 0]], 2.0, epsilon = 1e-14);
    assert_abs_diff_eq!(r[[1, 1]], 4.0, epsilon = 1e-14);
}

/// Two snapshots [1,0]ᵀ and [0,1]ᵀ average to the identity matrix divided by 1.
///
/// R = ([1,0]·[1,0]ᵀ + [0,1]·[0,1]ᵀ) / 2 = ([[1,0],[0,0]] + [[0,0],[0,1]]) / 2
///   = [[0.5, 0], [0, 0.5]].
#[test]
fn estimate_two_orthogonal_snapshots_gives_half_identity() {
    let mut x = Array2::<f64>::zeros((2, 2));
    x[[0, 0]] = 1.0; // snapshot 0: [1, 0]
    x[[1, 1]] = 1.0; // snapshot 1: [0, 1]

    let est = CovarianceEstimator {
        forward_backward_averaging: false,
        num_snapshots: 2,
        post_process: CovariancePostProcess::None,
    };
    let r = est.estimate(&x).unwrap();

    assert_abs_diff_eq!(r[[0, 0]], 0.5, epsilon = 1e-14);
    assert_abs_diff_eq!(r[[1, 1]], 0.5, epsilon = 1e-14);
    assert_abs_diff_eq!(r[[0, 1]], 0.0, epsilon = 1e-14);
    assert_abs_diff_eq!(r[[1, 0]], 0.0, epsilon = 1e-14);
}

#[test]
fn spatial_smoothing_complex_shapes_match() {
    let mut r = Array2::<Complex64>::from_elem((4, 4), Complex64::default());
    for i in 0..4 {
        r[[i, i]] = Complex64::new(1.0 + i as f64, 0.0);
    }

    let smoother = SpatialSmoothingComplex::new(3);
    let sm = smoother.apply(&r).expect("smoothed");
    assert_eq!(sm.shape()[0], 3);
    assert_eq!(sm.shape()[1], 3);
}
