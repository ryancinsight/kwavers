use super::ext::LinearAlgebraExt;
use crate::linear_algebra::ext::norm_l2;
use eunomia::Complex64 as Complex;
use leto::{Array1, Array2, Array3};
use leto_ops::{solve, symmetric_eigenvalues_jacobi};

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
    let array =
        Array3::from_vec([2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let norm = norm_l2(&array);
    let expected = (1..=8).map(|x| (x * x) as f64).sum::<f64>().sqrt();
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn test_linear_algebra_ext_trait() {
    let a = Array2::from_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array1::from_vec(2, vec![5.0, 11.0]).unwrap();

    let x = a.solve_into(b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-6);
    assert!((x[1] - 2.0).abs() < 1e-6);
}

#[test]
fn complex_ext_eig_delegates_to_hermitian_solver() {
    let matrix = Array2::from_vec(
        [2, 2],
        vec![
            Complex::new(2.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(1.0, 1.0),
            Complex::new(3.0, 0.0),
        ],
    )
    .unwrap();

    let (eigenvalues, eigenvectors) = matrix.eig().unwrap();

    assert!((eigenvalues[0] - Complex::new(4.0, 0.0)).norm() < 1e-10);
    assert!((eigenvalues[1] - Complex::new(1.0, 0.0)).norm() < 1e-10);

    for column in 0..2 {
        let lambda = eigenvalues[column];
        let vector = eigenvectors
            .index_axis::<1>(1, column)
            .unwrap()
            .to_contiguous();
        let residual: Vec<Complex> = (0..2)
            .map(|i| {
                let av = (0..2).map(|j| matrix[[i, j]] * vector[j]).sum::<Complex>();
                av - lambda * vector[i]
            })
            .collect();
        assert!(residual.iter().all(|entry| entry.norm() < 1e-10));
    }
}

#[test]
fn complex_ext_eig_rejects_non_hermitian_matrix() {
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

    let error = matrix.eig().unwrap_err();
    assert!(format!("{error}").contains("not Hermitian"));
}

#[test]
fn eigendecomposition_symmetric_2x2() {
    let a = Array2::<f64>::from_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let (vals2, vecs) = a.eig().unwrap();

    // Cross-check the eigenvalue set against an independent oracle. The oracle
    // (`leto_ops`) sorts ascending while `eig()` sorts descending, so compare
    // as order-independent sets rather than element-wise.
    let oracle = symmetric_eigenvalues_jacobi(&a.view()).unwrap();
    let mut oracle_sorted = oracle.clone();
    oracle_sorted.sort_by(|x, y| y.total_cmp(x));
    for (computed, expected) in (0..vals2.len()).map(|i| vals2[i]).zip(oracle_sorted) {
        assert!((computed - expected).abs() < 1e-10);
    }

    // Authoritative check: each returned (λ_i, v_i) pair satisfies A·v = λ·v.
    for i in 0..vals2.len() {
        let lambda = vals2[i];
        let v = vecs.index_axis::<1>(1, i).unwrap().to_contiguous();
        let mut av = Array1::<f64>::zeros(2);
        leto_ops::matvec(&a.view(), &v.view(), &mut av.view_mut()).unwrap();
        let lv = v.mapv(|x| lambda * x);
        assert!(av.iter().zip(lv.iter()).all(|(a, b)| (a - b).abs() < 1e-10));
    }
}
