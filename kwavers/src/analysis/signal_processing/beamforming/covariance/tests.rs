use super::{
    estimate_forward_backward_covariance, estimate_sample_covariance, is_hermitian, trace,
    validate_covariance_matrix,
};
use approx::assert_relative_eq;
use ndarray::Array2;
use num_complex::Complex64;

#[test]
fn test_sample_covariance_basic() {
    let mut data = Array2::<Complex64>::zeros((2, 10));
    for m in 0..10 {
        data[[0, m]] = Complex64::new(1.0, 0.0);
        data[[1, m]] = Complex64::new(0.5, 0.5);
    }

    let cov = estimate_sample_covariance(&data, 0.0).expect("should compute");

    assert_eq!(cov.shape(), &[2, 2]);
    assert!(is_hermitian(&cov, 1e-10));
    assert!(cov[[0, 0]].im.abs() < 1e-10);
    assert!(cov[[1, 1]].im.abs() < 1e-10);
    assert!(cov[[0, 0]].re > 0.0);
    assert!(cov[[1, 1]].re > 0.0);
}

#[test]
fn test_sample_covariance_with_diagonal_loading() {
    let data = Array2::<Complex64>::zeros((4, 8));
    let loading = 1e-3;

    let cov = estimate_sample_covariance(&data, loading).expect("should compute");

    for i in 0..4 {
        assert_relative_eq!(cov[[i, i]].re, loading, epsilon = 1e-10);
        assert_relative_eq!(cov[[i, i]].im, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_sample_covariance_insufficient_snapshots() {
    let data = Array2::<Complex64>::zeros((8, 4));

    let result = estimate_sample_covariance(&data, 0.0);
    assert!(result.is_err());

    let cov = estimate_sample_covariance(&data, 1e-4).unwrap();
    assert_eq!(
        cov.dim(),
        (8, 8),
        "covariance matrix must be 8×8 (n_sensors)"
    );
}

#[test]
fn test_sample_covariance_invalid_inputs() {
    let data = Array2::<Complex64>::zeros((4, 0));
    assert!(estimate_sample_covariance(&data, 0.0).is_err());

    let data = Array2::<Complex64>::zeros((4, 8));
    assert!(estimate_sample_covariance(&data, -1.0).is_err());
    assert!(estimate_sample_covariance(&data, f64::NAN).is_err());
}

#[test]
fn test_forward_backward_averaging() {
    let mut data = Array2::<Complex64>::zeros((4, 10));
    for m in 0..10 {
        for i in 0..4 {
            data[[i, m]] = Complex64::new((i as f64) * 0.1, (m as f64) * 0.01);
        }
    }

    let cov_fb = estimate_forward_backward_covariance(&data, 1e-4).expect("should compute");

    assert!(is_hermitian(&cov_fb, 1e-10));
    assert_eq!(cov_fb.shape(), &[4, 4]);
    validate_covariance_matrix(&cov_fb).expect("should be valid");
}

#[test]
fn test_is_hermitian() {
    let mut matrix = Array2::<Complex64>::zeros((3, 3));

    matrix[[0, 0]] = Complex64::new(1.0, 0.0);
    matrix[[1, 1]] = Complex64::new(2.0, 0.0);
    matrix[[2, 2]] = Complex64::new(3.0, 0.0);

    matrix[[0, 1]] = Complex64::new(0.5, 0.2);
    matrix[[1, 0]] = Complex64::new(0.5, -0.2);

    matrix[[0, 2]] = Complex64::new(0.3, -0.1);
    matrix[[2, 0]] = Complex64::new(0.3, 0.1);

    matrix[[1, 2]] = Complex64::new(0.1, 0.3);
    matrix[[2, 1]] = Complex64::new(0.1, -0.3);

    assert!(is_hermitian(&matrix, 1e-10));

    matrix[[0, 1]] = Complex64::new(0.5, 0.3);
    assert!(!is_hermitian(&matrix, 1e-10));
}

#[test]
fn test_trace() {
    let mut matrix = Array2::<Complex64>::zeros((3, 3));
    matrix[[0, 0]] = Complex64::new(1.0, 0.0);
    matrix[[1, 1]] = Complex64::new(2.0, 0.0);
    matrix[[2, 2]] = Complex64::new(3.0, 0.0);

    let tr = trace(&matrix).expect("should compute");
    assert_relative_eq!(tr.re, 6.0, epsilon = 1e-10);
    assert_relative_eq!(tr.im, 0.0, epsilon = 1e-10);
}

#[test]
fn test_validate_covariance_matrix() {
    let data = Array2::<Complex64>::zeros((4, 20));
    let cov = estimate_sample_covariance(&data, 1e-4).expect("should compute");
    validate_covariance_matrix(&cov).expect("should be valid");

    let non_square = Array2::<Complex64>::zeros((3, 4));
    assert!(validate_covariance_matrix(&non_square).is_err());

    let mut non_hermitian = Array2::<Complex64>::zeros((3, 3));
    non_hermitian[[0, 1]] = Complex64::new(1.0, 1.0);
    non_hermitian[[1, 0]] = Complex64::new(1.0, 0.0);
    assert!(validate_covariance_matrix(&non_hermitian).is_err());
}

#[test]
fn test_covariance_with_non_finite_data() {
    let mut data = Array2::<Complex64>::zeros((4, 10));
    data[[0, 0]] = Complex64::new(f64::NAN, 0.0);

    assert!(estimate_sample_covariance(&data, 0.0).is_err());
}

// ─── Exact value-semantic tests ────────────────────────────────────────────────

/// `estimate_sample_covariance` on rank-1 data with known cross terms.
///
/// 2 sensors, M=4 snapshots, each = [1+0j, 0+1j]^T.
/// R = (1/M) Σ x x^H:
///   R[0,0] = |1|² = 1.0,  R[0,1] = (1)·(0−1j) = 0−j,
///   R[1,0] = (0+1j)·(1)^* = 0+j,  R[1,1] = |j|² = 1.0.
#[test]
fn covariance_sample_rank1_exact_cross_terms() {
    let mut data = Array2::<Complex64>::zeros((2, 4));
    for m in 0..4 {
        data[[0, m]] = Complex64::new(1.0, 0.0);
        data[[1, m]] = Complex64::new(0.0, 1.0);
    }
    let r = estimate_sample_covariance(&data, 0.0).expect("should succeed");
    assert!(
        (r[[0, 0]].re - 1.0).abs() < 1e-14 && r[[0, 0]].im.abs() < 1e-14,
        "R[0,0] = {:?} (expected 1+0j)",
        r[[0, 0]]
    );
    assert!(
        (r[[1, 1]].re - 1.0).abs() < 1e-14 && r[[1, 1]].im.abs() < 1e-14,
        "R[1,1] = {:?} (expected 1+0j)",
        r[[1, 1]]
    );
    assert!(
        r[[0, 1]].re.abs() < 1e-14 && (r[[0, 1]].im + 1.0).abs() < 1e-14,
        "R[0,1] = {:?} (expected 0−j)",
        r[[0, 1]]
    );
    assert!(
        r[[1, 0]].re.abs() < 1e-14 && (r[[1, 0]].im - 1.0).abs() < 1e-14,
        "R[1,0] = {:?} (expected 0+j)",
        r[[1, 0]]
    );
}

/// `estimate_sample_covariance` with diagonal loading shifts diagonal exactly.
///
/// All-zero data + loading=0.3 → R[i,i] = 0.3, off-diagonals = 0.
#[test]
fn covariance_sample_diagonal_loading_exact() {
    let data = Array2::<Complex64>::zeros((3, 6));
    let loading = 0.3_f64;
    let r = estimate_sample_covariance(&data, loading).expect("should succeed");
    for i in 0..3 {
        assert!(
            (r[[i, i]].re - loading).abs() < 1e-14,
            "R[{i},{i}].re = {} (expected {loading})",
            r[[i, i]].re
        );
        assert!(r[[i, i]].im.abs() < 1e-14, "R[{i},{i}].im must be 0");
    }
    assert!(
        r[[0, 1]].norm() < 1e-14,
        "off-diagonal must be 0 for zero data"
    );
}

/// `trace` of a 2×2 diagonal matrix gives exact sum of diagonal.
///
/// R = diag(1+2j, 3−j) → tr = (1+2j)+(3−j) = 4+j.
#[test]
fn covariance_trace_diagonal_matrix_exact() {
    let mut r = Array2::<Complex64>::zeros((2, 2));
    r[[0, 0]] = Complex64::new(1.0, 2.0);
    r[[1, 1]] = Complex64::new(3.0, -1.0);
    let tr = trace(&r).expect("trace should succeed");
    assert!(
        (tr.re - 4.0).abs() < 1e-14,
        "tr.re = {} (expected 4.0)",
        tr.re
    );
    assert!(
        (tr.im - 1.0).abs() < 1e-14,
        "tr.im = {} (expected 1.0)",
        tr.im
    );
}

/// `trace` on non-square matrix returns Err.
#[test]
fn covariance_trace_non_square_returns_err() {
    let r = Array2::<Complex64>::zeros((2, 3));
    assert!(
        trace(&r).is_err(),
        "trace of non-square matrix must return Err"
    );
}

/// `is_hermitian` detects exact conjugate symmetry.
///
/// R = [[2, 1+j], [1−j, 3]] is Hermitian; flipping one entry breaks it.
#[test]
fn covariance_is_hermitian_exact() {
    let mut r = Array2::<Complex64>::zeros((2, 2));
    r[[0, 0]] = Complex64::new(2.0, 0.0);
    r[[1, 1]] = Complex64::new(3.0, 0.0);
    r[[0, 1]] = Complex64::new(1.0, 1.0);
    r[[1, 0]] = Complex64::new(1.0, -1.0); // conjugate

    assert!(is_hermitian(&r, 1e-14), "R must be Hermitian");

    r[[1, 0]] = Complex64::new(1.0, 1.0); // break conjugate symmetry
    assert!(
        !is_hermitian(&r, 1e-14),
        "broken conjugate symmetry must fail is_hermitian"
    );
}

/// `estimate_forward_backward_covariance` on a single-sensor case equals the input.
///
/// For N=1: J = [[1]], R_fb = (1/2)(R_f + R_f^*·J·J) = (1/2)(R_f + R_f^*).
/// If data is real-valued (1 sensor, many snapshots), R_f is real scalar → R_fb = R_f.
#[test]
fn covariance_forward_backward_single_sensor_matches_sample() {
    let mut data = Array2::<Complex64>::zeros((1, 4));
    for m in 0..4 {
        data[[0, m]] = Complex64::new((m + 1) as f64, 0.0);
    }
    // sample covariance with loading=1e-4 (avoids insufficient-snapshots error for N=1)
    let r_sample = estimate_sample_covariance(&data, 1e-4).expect("should succeed");
    let r_fb = estimate_forward_backward_covariance(&data, 1e-4).expect("should succeed");
    // For 1 sensor: FB averaging must equal the sample covariance
    assert!(
        (r_fb[[0, 0]].re - r_sample[[0, 0]].re).abs() < 1e-12,
        "FB[0,0].re={} sample[0,0].re={}",
        r_fb[[0, 0]].re,
        r_sample[[0, 0]].re
    );
}
