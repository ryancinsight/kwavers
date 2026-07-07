use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::{Array1, Array2};
use leto_ops::{
    qr_decompose, svd_rank_revealing, MatrixSolve,
};
use std::fmt::Display;

/// Basic linear algebra operations for real-valued matrices
#[derive(Debug)]
pub struct LinearAlgebra;

impl LinearAlgebra {
    /// Solve a linear system Ax = b using LU decomposition.
    pub fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        let n = a.shape()[0];
        if a.shape()[1] != n || b.shape()[0] != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "solve_linear_system".to_owned(),
                expected: format!("{n}×{n} matrix and {n} vector"),
                actual: format!(
                    "{}×{} matrix and {} vector",
                    a.shape()[0],
                    a.shape()[1],
                    b.shape()[0]
                ),
            }));
        }
        a.solve(&b.view())
            .map_err(|e| linalg_error("LU solve", e))
    }

    /// Compute the inverse of a square matrix via LU decomposition.
    pub fn matrix_inverse(matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = matrix.shape()[0];
        if matrix.shape()[1] != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "matrix_inverse".to_owned(),
                expected: format!("{n}×{n} square matrix"),
                actual: format!("{}×{} matrix", matrix.shape()[0], matrix.shape()[1]),
            }));
        }
        matrix.inv().map_err(|e| linalg_error("LU inverse", e))
    }

    /// Compute the Householder QR decomposition.
    pub fn qr_decomposition(
        matrix: &Array2<f64>,
    ) -> KwaversResult<(Array2<f64>, Array2<f64>)> {
        let qr = qr_decompose(&matrix.view())
            .map_err(|e| linalg_error("QR decomposition", e))?;
        Ok((qr.q().to_owned(), qr.r().to_owned()))
    }

    /// Compute SVD decomposition of a matrix.
    pub fn svd(
        matrix: &Array2<f64>,
    ) -> KwaversResult<(Array2<f64>, Vec<f64>, Array2<f64>)> {
        let svd = svd_rank_revealing(&matrix.view())
            .map_err(|e| linalg_error("SVD", e))?;
        Ok((svd.left_singular_vectors, svd.singular_values, svd.right_singular_vectors))
    }
}

fn linalg_error(method: &'static str, error: impl Display) -> KwaversError {
    KwaversError::Numerical(NumericalError::SolverFailed {
        method: method.to_owned(),
        reason: error.to_string(),
    })
}

    #[cfg(test)]
    mod tests {
        use super::*;
        use leto::Storage;

        #[test]
    fn test_solve_linear_system() {
        let a = Array2::from_shape_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Array1::from(vec![3.0, 3.0]);

        let x = LinearAlgebra::solve_linear_system(&a, &b).unwrap();
        let xs = x.storage().as_slice();
        assert!((xs[0] - 1.0).abs() < 1e-10);
        assert!((xs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let a_inv = LinearAlgebra::matrix_inverse(&a).unwrap();

        let mut ok = true;
        for i in 0..2 {
            for j in 0..2 {
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                let got;
                {
                    let a_sl = a.storage().as_slice();
                    let ai_sl = a_inv.storage().as_slice();
                    let n = a.shape()[0];
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += a_sl[i * n + k] * ai_sl[k * n + j];
                    }
                    got = sum;
                }
                if (got - expected).abs() >= 1e-10 {
                    ok = false;
                }
            }
        }
        assert!(ok, "A * A_inv != I");
    }

    #[test]
    fn test_qr_reconstruction_and_orthogonality() {
        let cases = [
            Array2::from_shape_vec(
                [3, 3],
                vec![1.0, 2.0, 0.0, 0.0, 1.0, 3.0, 4.0, 0.0, 1.0],
            )
            .unwrap(),
            Array2::from_shape_vec([4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap(),
        ];
        for a in &cases {
            let (q, r) = LinearAlgebra::qr_decomposition(a).unwrap();
            let a_sl = a.storage().as_slice();
            let q_sl = q.storage().as_slice();
            let r_sl = r.storage().as_slice();
            let m = a.shape()[0];
            let n = a.shape()[1];

            // A = Q·R
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..m.min(n) {
                        sum += q_sl[i * m + k] * r_sl[k * n + j];
                    }
                    assert!(
                        (sum - a_sl[i * n + j]).abs() < 1e-9,
                        "QR reconstruction mismatch at [{i},{j}]"
                    );
                }
            }

            // Qᵀ Q = I
            for i in 0..m {
                for j in 0..m {
                    let mut sum = 0.0;
                    for k in 0..m {
                        sum += q_sl[k * m + i] * q_sl[k * m + j];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((sum - expected).abs() < 1e-9, "Q not orthonormal");
                }
            }

            // R is upper triangular
            for i in 0..m {
                for j in 0..i.min(n) {
                    assert!(r_sl[i * n + j].abs() < 1e-9, "R not upper triangular at [{i},{j}]");
                }
            }
        }
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let a_sl = a.storage().as_slice();
        let (u, s, v) = LinearAlgebra::svd(&a).unwrap();
        let u_sl = u.storage().as_slice();
        let v_sl = v.storage().as_slice();

        // A = U·S·Vᵀ
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u_sl[i * 2 + k] * s[k] * v_sl[j * 2 + k];
                }
                assert!((sum - a_sl[i * 2 + j]).abs() < 1e-10);
            }
        }

        // Uᵀ U = I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u_sl[k * 2 + i] * u_sl[k * 2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 1e-10, "U not orthogonal");
            }
        }

        // Vᵀ V = I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += v_sl[k * 2 + i] * v_sl[k * 2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 1e-10, "V not orthogonal");
            }
        }
    }
}
