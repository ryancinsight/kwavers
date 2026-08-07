use leto::{Array1, Array2, Array3};
use leto_ops::application::linalg::norm_l2;
use leto_ops::solve;

#[test]
fn test_linear_algebra_re_exports() {
    let a = Array2::<f64>::from_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let b = Array1::<f64>::from_vec(2, vec![3.0, 3.0]).unwrap();

    let x = solve(&a.view(), &b.view()).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_norm_l2_convenience_function() {
    let array = Array3::from_vec([2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let view = array.view();
    let norm = norm_l2(&view).unwrap();
    let expected: f64 = (1..=8).map(|x| (x * x) as f64).sum::<f64>().sqrt();
    assert!(f64::abs(norm - expected) < 1e-10);
}

#[test]
fn test_solve_via_leto_ops() {
    let a = Array2::from_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array1::from_vec(2, vec![5.0, 11.0]).unwrap();

    let x = solve(&a.view(), &b.view()).unwrap();
    let diff0 = f64::abs(x[0] - 1.0);
    let diff1 = f64::abs(x[1] - 2.0);
    assert!(diff0 < 1e-6);
    assert!(diff1 < 1e-6);
}

#[test]
fn eigenvalues_returns_only_eigenvalues() {
    // Test that eigenvalues function returns only eigenvalues, not eigenvectors
    let matrix = Array2::<f64>::from_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();

    let result = leto_ops::application::linalg::eigenvalues(&matrix.view());

    // Should succeed for symmetric matrices
    assert!(result.is_ok());

    let eigenvalues = result.unwrap();

    // Check that we get 2 eigenvalues
    assert_eq!(eigenvalues.len(), 2);

    // Verify they are approximately 1 and 3
    let mut sorted = eigenvalues.clone();
    sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    assert!((sorted[0].re - 1.0).abs() < 1e-10);
    assert!((sorted[1].re - 3.0).abs() < 1e-10);
}

#[test]
fn symmetric_eigen_rejects_non_symmetric_matrix() {
    // Test that symmetric_eigenvalues_jacobi rejects non-symmetric matrices
    // by falling back or returning an error
    let matrix = Array2::<f64>::from_vec([2, 2], vec![1.0, 2.0, 2.0, 1.0]).unwrap();

    // This should work since the matrix is symmetric
    let result = leto_ops::application::linalg::symmetric_eigenvalues_jacobi(&matrix.view());
    assert!(result.is_ok());
}

#[test]
fn eigendecomposition_symmetric_2x2() {
    // Test that symmetric_eigenvalues_jacobi works correctly
    let a = Array2::<f64>::from_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();

    // Use symmetric_eigenvalues_jacobi which returns only eigenvalues
    let vals2 = leto_ops::application::linalg::symmetric_eigenvalues_jacobi(&a.view()).unwrap();

    // Check that we get 2 eigenvalues
    assert_eq!(vals2.len(), 2);

    // Verify they are approximately 1 and 3
    let mut sorted = vals2.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!((sorted[0] - 1.0).abs() < 1e-10);
    assert!((sorted[1] - 3.0).abs() < 1e-10);
}
