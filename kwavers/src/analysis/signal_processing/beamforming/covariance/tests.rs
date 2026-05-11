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
    assert_eq!(cov.dim(), (8, 8), "covariance matrix must be 8×8 (n_sensors)");
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
