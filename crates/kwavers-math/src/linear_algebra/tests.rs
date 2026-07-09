use super::ext::LinearAlgebraExt;
use super::{EigenDecomposition, LinearAlgebra};
use crate::linear_algebra::ext::norm_l2;
use eunomia::Complex64 as Complex;
use leto::{Array1, Array2, Array3};

#[test]
fn test_linear_algebra_re_exports() {
    let a = Array2::from_vec(vec![2.0, 1.0, 1.0, 2.0], (2, 2)).unwrap();
    let b = Array1::from_vec(vec![3.0, 3.0]);

    let x = LinearAlgebra::solve_linear_system(&a, &b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_norm_l2_convenience_function() {
    let array =
        Array3::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 2, 2))
            .unwrap();
    let norm = norm_l2(&array);
    let expected = (1..=8).map(|x| (x * x) as f64).sum::<f64>().sqrt();
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn test_linear_algebra_ext_trait() {
    let a = Array2::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).unwrap();
    let b = Array1::from_vec(vec![5.0, 11.0]);

    let x = a.solve_into(b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-6);
    assert!((x[1] - 2.0).abs() < 1e-6);
}

#[test]
fn complex_ext_eig_delegates_to_hermitian_solver() {
    let matrix = Array2::from_vec(
        vec![
            Complex::new(2.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(1.0, 1.0),
            Complex::new(3.0, 0.0),
        ],
        (2, 2),
    )
    .unwrap();

    let (eigenvalues, eigenvectors) = matrix.eig().unwrap();

    assert!((eigenvalues[0] - Complex::new(4.0, 0.0)).norm() < 1e-10);
    assert!((eigenvalues[1] - Complex::new(1.0, 0.0)).norm() < 1e-10);

    for column in 0..2 {
        let lambda = eigenvalues[column];
        let vector = eigenvectors.column(column).to_owned();
        let residual = matrix.matmul(&vector) - vector.mapv(|entry| lambda * entry);
        assert!(residual.iter().all(|entry| entry.norm() < 1e-10));
    }
}

#[test]
fn complex_ext_eig_rejects_non_hermitian_matrix() {
    let matrix = Array2::from_vec(
        vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 1.0),
            Complex::new(3.0, 0.0),
        ],
        (2, 2),
    )
    .unwrap();

    let error = matrix.eig().unwrap_err();
    assert!(format!("{error}").contains("not Hermitian"));
}

#[test]
fn eigendecomposition_symmetric_2x2() {
    let a = Array2::from_vec(vec![2.0, 1.0, 1.0, 2.0], (2, 2)).unwrap();
    let (vals, vecs) = EigenDecomposition::eigendecomposition(&a).unwrap();
    for (i, &lambda) in vals.iter().enumerate() {
        let v = vecs.column(i).to_owned();
        let av = a.matmul(&v);
        let lv = v.mapv(|x| lambda * x);
        assert!(av.iter().zip(lv.iter()).all(|(a, b)| (a - b).abs() < 1e-10));
    }
}
