use eunomia::Complex64 as Complex;
use leto::{Array1, Array2, Array3};
use leto_ops::{solve, symmetric_eigen_jacobi};

#[test]
fn test_linear_algebra_re_exports() {
    let a = Array2::<f64>::from_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let b = Array1::<f64>::from_vec(2, vec![3.0, 3.0]).unwrap();

    let x = solve(&a.view(), &b.view()).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_norm_l2_delegates_to_ssot() {
    let array = Array3::from_vec([2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let norm = leto_ops::norm_l2(&array.view()).unwrap();
    let expected = (1..=8).map(|x| (x * x) as f64).sum::<f64>().sqrt();
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn eigendecomposition_symmetric_2x2() {
    let a = Array2::<f64>::from_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let result = leto_ops::symmetric_eigen_jacobi(&a.view()).unwrap();
    let (eigenvalues, eigenvectors) = (result.eigenvalues, result.eigenvectors);

    // Cross-check the eigenvalue set against an independent oracle.
    let oracle = symmetric_eigen_jacobi(&a.view()).unwrap();
    let mut oracle_sorted = oracle.eigenvalues.clone();
    oracle_sorted.sort_by(|x, y| y.total_cmp(x));
    for (computed, expected) in (0..eigenvalues.len()).map(|i| eigenvalues[i]).zip(oracle_sorted) {
        assert!((computed - expected).abs() < 1e-10);
    }

    // Authoritative check: each returned (λ_i, v_i) pair satisfies A·v = λ·v.
    for i in 0..eigenvalues.len() {
        let lambda = eigenvalues[i];
        let v = eigenvectors.index_axis::<1>(1, i).unwrap().to_contiguous();
        let mut av = Array1::<f64>::zeros(2);
        leto_ops::matvec(&a.view(), &v.view(), &mut av.view_mut()).unwrap();
        let lv = v.mapv(|x| lambda * x);
        assert!(av.iter().zip(lv.iter()).all(|(a, b)| (a - b).abs() < 1e-10));
    }
}

#[test]
fn complex_solve_delegates_to_ssot() {
    let a = Array2::from_vec(
        [2, 2],
        vec![
            Complex::new(2.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(1.0, 1.0),
            Complex::new(3.0, 0.0),
        ],
    )
    .unwrap();
    let b = Array1::from_vec(2, vec![3.0, 3.0]).unwrap();

    let x = leto_ops::complex_solve(&a, &b).unwrap();

    assert!((x[0] - Complex::new(1.0, 0.0)).norm() < 1e-10);
    assert!((x[1] - Complex::new(1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn complex_solve_rejects_non_hermitian_matrix() {
    let matrix = Array2::from_vec(
        [2, 2],
        vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 1.0),
            Complex::new(3.0, 0.0),
        ],
    )
    .unwrap();
    let b = Array1::from_vec(2, vec![1.0, 2.0]).unwrap();

    let error = leto_ops::complex_solve(&matrix, &b).unwrap_err();
    assert!(format!("{error}").contains("not Hermitian"));
}
