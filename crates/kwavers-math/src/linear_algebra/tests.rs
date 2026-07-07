use super::ext::LinearAlgebraExt;
use super::{EigenDecomposition, LinearAlgebra};
use crate::linear_algebra::ext::norm_l2;
use leto::{Array1, Array2, Array3};
use leto_ops::MatrixProduct;
use num_complex::Complex;

#[test]
fn test_linear_algebra_re_exports() {
    let a = Array2::from_shape_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let b = Array1::from(vec![3.0, 3.0]);

    let x = LinearAlgebra::solve_linear_system(&a, &b).unwrap();
    assert!((x[[0]] - 1.0).abs() < 1e-10);
    assert!((x[[1]] - 1.0).abs() < 1e-10);
}

#[test]
fn test_norm_l2_convenience_function() {
    let array = Array3::from_shape_vec([2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap();
    let norm = norm_l2(&array);
    let expected = (1..=8).map(|x| (x * x) as f64).sum::<f64>().sqrt();
    assert!((norm - expected).abs() < 1e-10);
}

#[test]
fn test_linear_algebra_ext_trait() {
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array1::from(vec![5.0, 11.0]);

    let x = a.solve_into(b).unwrap();
    assert!((x[[0]] - 1.0).abs() < 1e-6);
    assert!((x[[1]] - 2.0).abs() < 1e-6);
}

fn extract_column<T: Copy>(matrix: &Array2<T>, col: usize) -> Array1<T> {
    let rows = matrix.shape()[0];
    Array1::from_shape_fn([rows], |[i]| matrix[[i, col]])
}

fn mat_vec_mul_complex(matrix: &Array2<Complex<f64>>, vector: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
    let rows = matrix.shape()[0];
    let cols = matrix.shape()[1];
    Array1::from_shape_fn([rows], |[i]| {
        let mut sum = Complex::new(0.0, 0.0);
        for j in 0..cols {
            sum = sum + matrix[[i, j]] * vector[[j]];
        }
        sum
    })
}

#[test]
fn complex_ext_eig_delegates_to_hermitian_solver() {
    let matrix = Array2::from_shape_vec(
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

    assert!((eigenvalues[[0]] - Complex::new(4.0, 0.0)).norm() < 1e-10);
    assert!((eigenvalues[[1]] - Complex::new(1.0, 0.0)).norm() < 1e-10);

    for column in 0..2 {
        let lambda = eigenvalues[[column]];
        let vector = extract_column(&eigenvectors, column);
        let mv = mat_vec_mul_complex(&matrix, &vector);
        let lv = vector.mapv(|entry| lambda * entry);
        for i in 0..mv.shape()[0] {
            assert!((mv[[i]] - lv[[i]]).norm() < 1e-10);
        }
    }
}

#[test]
fn complex_ext_eig_rejects_non_hermitian_matrix() {
    let matrix = Array2::from_shape_vec(
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
    let a = Array2::from_shape_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let (vals, vecs) = EigenDecomposition::eigendecomposition(&a).unwrap();
    for i in 0..vals.shape()[0] {
        let lambda = vals[[i]];
        let v = extract_column(&vecs, i);
        let v_mat = Array2::from_shape_fn([2, 1], |[r, _]| v[[r]]);
        let av = a.matmul(&v_mat).unwrap();
        let lv = v_mat.mapv(|x| lambda * x);
        assert!(av.iter().zip(lv.iter()).all(|(a, b)| (a - b).abs() < 1e-10));
    }
}
