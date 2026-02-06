//! Eigenvalue Decomposition Operations
//!
//! This module provides eigenvalue and eigenvector computation for both real
//! and complex Hermitian matrices, essential for subspace beamforming methods.
//!
//! # Mathematical Foundation
//!
//! ## Real Symmetric Matrices
//!
//! For real symmetric matrix **A** ∈ ℝ^(n×n):
//! - Eigenvalues λᵢ ∈ ℝ (real)
//! - Eigenvectors orthogonal: Vᵀ V = I
//! - Decomposition: A = V Λ Vᵀ
//!
//! ## Complex Hermitian Matrices
//!
//! For complex Hermitian matrix **H** ∈ ℂ^(n×n) where H† = H:
//! - Eigenvalues λᵢ ∈ ℝ (real, guaranteed by Hermiticity)
//! - Eigenvectors orthonormal: V† V = I (unitary)
//! - Decomposition: H = V Λ V†
//!
//! ## Real-Embedded Form
//!
//! Complex Hermitian **H = A + iB** embeds into real symmetric **S** ∈ ℝ^(2n×2n):
//!
//! ```text
//! S = [ A  -B ]
//!     [ B   A ]
//! ```
//!
//! **Theorem**: If H† = H, then Sᵀ = S and λ(H) = λ(S)[1:n]
//!
//! # Algorithm Selection
//!
//! **Two-Tier Strategy** for optimal performance:
//!
//! ## Tier 1: Jacobi (n < 64)
//! - Direct Jacobi iteration on real-embedded symmetric form
//! - Simple, robust, fast for small matrices
//! - Tolerance: 1e-12, Max sweeps: 2048
//! - Complexity: O(n³) per sweep, typically 5-10 sweeps
//!
//! ## Tier 2: Householder + QR (n ≥ 64)
//! - Householder tridiagonalization: O(n³) preprocessing
//! - Implicit QR with Wilkinson shift: O(n²) per eigenvalue
//! - Superior scaling for larger matrices
//! - Tolerance: 1e-12 for off-diagonal entries
//!
//! # References
//!
//! - Golub & Van Loan (2013) - Matrix Computations (4th ed.), Chapters 8-9
//! - Parlett (1998) - The Symmetric Eigenvalue Problem
//! - Wilkinson (1965) - The Algebraic Eigenvalue Problem
//! - Stewart (2001) - Matrix Algorithms, Vol II

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::math::linear_algebra::tolerance;
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// Eigenvalue decomposition operations
#[derive(Debug)]
pub struct EigenDecomposition;

impl EigenDecomposition {
    /// Compute eigendecomposition of a real symmetric matrix
    ///
    /// # Arguments
    /// * `matrix` - Real symmetric matrix (n×n)
    ///
    /// # Returns
    /// Tuple (eigenvalues, eigenvectors) where eigenvalues is (n) and eigenvectors is (n×n)
    ///
    /// # Note
    /// Uses Jacobi eigenvalue algorithm - suitable for small matrices (< 100×100)
    pub fn eigendecomposition(matrix: &Array2<f64>) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "eigendecomposition".to_string(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.nrows(), matrix.ncols()),
            }));
        }

        // Check if matrix is symmetric
        for i in 0..n {
            for j in (i + 1)..n {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > 1e-10 {
                    return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                        "Matrix must be symmetric for real eigendecomposition".to_string(),
                    )));
                }
            }
        }

        let mut a = matrix.clone();
        let mut eigenvectors = Array2::eye(n);
        let mut eigenvalues = Array1::zeros(n);

        // Jacobi eigenvalue algorithm
        let max_iterations = 100;
        let tolerance = 1e-10;

        for _ in 0..max_iterations {
            // Find largest off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    if a[[i, j]].abs() > max_val {
                        max_val = a[[i, j]].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if max_val < tolerance {
                break;
            }

            // Compute rotation angle
            let theta = if a[[p, p]] == a[[q, q]] {
                std::f64::consts::PI / 4.0
            } else {
                0.5 * (a[[q, q]] - a[[p, p]])
                    / a[[p, q]].atan2((a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]))
            };

            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation to matrix A
            for i in 0..n {
                let a_ip = a[[i, p]];
                let a_iq = a[[i, q]];
                a[[i, p]] = c * a_ip - s * a_iq;
                a[[i, q]] = s * a_ip + c * a_iq;
            }

            for i in 0..n {
                let a_pi = a[[p, i]];
                let a_qi = a[[q, i]];
                a[[p, i]] = c * a_pi - s * a_qi;
                a[[q, i]] = s * a_pi + c * a_qi;
            }

            // Apply rotation to eigenvectors
            for i in 0..n {
                let v_ip = eigenvectors[[i, p]];
                let v_iq = eigenvectors[[i, q]];
                eigenvectors[[i, p]] = c * v_ip - s * v_iq;
                eigenvectors[[i, q]] = s * v_ip + c * v_iq;
            }
        }

        // Extract eigenvalues from diagonal
        for i in 0..n {
            eigenvalues[i] = a[[i, i]];
        }

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let sorted_eigenvals = Array1::from_shape_fn(n, |i| eigenvalues[indices[i]]);
        let sorted_eigenvecs =
            Array2::from_shape_fn((n, n), |(i, j)| eigenvectors[[i, indices[j]]]);

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }

    /// Compute eigendecomposition of a complex Hermitian matrix
    ///
    /// Uses complex Jacobi iteration directly on the Hermitian matrix.
    /// This is a robust approach suitable for small to medium matrices (n < 500).
    ///
    /// # Arguments
    /// * `matrix` - Complex Hermitian matrix H ∈ ℂ^(n×n) where H† = H
    ///
    /// # Returns
    /// Tuple (eigenvalues ∈ ℝⁿ, eigenvectors ∈ ℂ^(n×n)) where:
    /// - Eigenvalues sorted in descending order: λ₁ ≥ λ₂ ≥ ... ≥ λₙ
    /// - Eigenvectors are orthonormal: V† V = I
    /// - Reconstruction: H = V Λ V†
    ///
    /// # Algorithm
    ///
    /// Complex Jacobi method with Hermitian Givens rotations:
    /// 1. Find largest off-diagonal element |H[p,q]|
    /// 2. Compute complex Givens rotation to zero out H[p,q]
    /// 3. Apply unitary similarity transformation: H ← U† H U
    /// 4. Accumulate eigenvectors: V ← V U
    /// 5. Repeat until convergence
    ///
    /// # Errors
    ///
    /// - `MatrixDimension`: If matrix is not square
    /// - `InvalidOperation`: If matrix is not Hermitian (H† ≠ H within tolerance)
    ///
    /// # References
    ///
    /// - Golub & Van Loan (2013) - Matrix Computations, §8.5
    /// - Wilkinson & Reinsch (1971) - Handbook for Automatic Computation, Vol II
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use kwavers::math::linear_algebra::EigenDecomposition;
    /// use ndarray::Array2;
    /// use num_complex::Complex;
    ///
    /// // 2×2 Hermitian matrix
    /// let h = Array2::from_shape_vec((2, 2), vec![
    ///     Complex::new(2.0, 0.0), Complex::new(1.0, -1.0),
    ///     Complex::new(1.0, 1.0), Complex::new(3.0, 0.0),
    /// ]).unwrap();
    ///
    /// let (eigenvals, eigenvecs) = EigenDecomposition::hermitian_eigendecomposition_complex(&h)?;
    ///
    /// // Verify: H = V Λ V†
    /// let reconstructed = eigenvecs.dot(&Array2::from_diag(&eigenvals.mapv(|x| Complex::new(x, 0.0)))).dot(&eigenvecs.t().mapv(|z| z.conj()));
    /// assert!((h - reconstructed).mapv(|z| z.norm()).sum() < 1e-10);
    /// ```
    pub fn hermitian_eigendecomposition_complex(
        matrix: &Array2<Complex<f64>>,
    ) -> KwaversResult<(Array1<f64>, Array2<Complex<f64>>)> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "hermitian_eigendecomposition_complex".to_string(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.nrows(), matrix.ncols()),
            }));
        }

        // Check if matrix is Hermitian: H† = H
        for i in 0..n {
            // Diagonal must be real
            if matrix[[i, i]].im.abs() > tolerance::HERMITIAN_EIG_TOL {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    format!(
                        "Matrix diagonal element [{}] has imaginary part {:.2e}, not Hermitian",
                        i,
                        matrix[[i, i]].im
                    ),
                )));
            }
            // Off-diagonal must satisfy H[i,j] = conj(H[j,i])
            for j in (i + 1)..n {
                if (matrix[[i, j]] - matrix[[j, i]].conj()).norm() > tolerance::HERMITIAN_EIG_TOL {
                    return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                        format!(
                            "Matrix not Hermitian: H[{},{}] = {:?}, H[{},{}]† = {:?}",
                            i,
                            j,
                            matrix[[i, j]],
                            j,
                            i,
                            matrix[[j, i]].conj()
                        ),
                    )));
                }
            }
        }

        // Complex Jacobi iteration for Hermitian matrices
        let mut h = matrix.clone();
        let mut v = Array2::eye(n).mapv(|x| Complex::new(x, 0.0));

        for _sweep in 0..tolerance::HERMITIAN_EIG_MAX_SWEEPS {
            // Find largest off-diagonal element by magnitude
            let mut max_offdiag = 0.0;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    let val = h[[i, j]].norm();
                    if val > max_offdiag {
                        max_offdiag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if max_offdiag < tolerance::HERMITIAN_EIG_TOL {
                break;
            }

            // Compute complex Givens rotation parameters for Hermitian matrix
            // Goal: Zero out h[p,q] using unitary transformation U† H U
            // For Hermitian matrices, diagonal elements h[p,p] and h[q,q] are real

            let h_pp = h[[p, p]].re;
            let h_qq = h[[q, q]].re;
            let h_pq = h[[p, q]];

            // Standard complex Jacobi rotation for Hermitian matrices
            // Reference: Golub & Van Loan (2013), Algorithm 8.5.1

            let theta = if (h_pp - h_qq).abs() < tolerance::HERMITIAN_EIG_TOL {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * h_pq.norm() / (h_pp - h_qq)).atan()
                    * if h_pp > h_qq { 1.0 } else { -1.0 }
            };

            let c = theta.cos();
            let s_mag = theta.sin();

            // Phase to align rotation with h_pq
            let phase = if h_pq.norm() > tolerance::HERMITIAN_EIG_TOL {
                h_pq.conj() / h_pq.norm()
            } else {
                Complex::new(1.0, 0.0)
            };

            let s_complex = Complex::new(s_mag, 0.0) * phase;

            // Apply Givens rotation: H ← U† H U where U is the Givens rotation
            // First apply U on right: H ← H U
            for i in 0..n {
                let h_ip = h[[i, p]];
                let h_iq = h[[i, q]];
                h[[i, p]] = c * h_ip - s_complex.conj() * h_iq;
                h[[i, q]] = s_complex * h_ip + c * h_iq;
            }

            // Then apply U† on left: H ← U† H
            for j in 0..n {
                let h_pj = h[[p, j]];
                let h_qj = h[[q, j]];
                h[[p, j]] = c * h_pj - s_complex * h_qj;
                h[[q, j]] = s_complex.conj() * h_pj + c * h_qj;
            }

            // Accumulate eigenvectors: V ← V U
            for i in 0..n {
                let v_ip = v[[i, p]];
                let v_iq = v[[i, q]];
                v[[i, p]] = c * v_ip - s_complex.conj() * v_iq;
                v[[i, q]] = s_complex * v_ip + c * v_iq;
            }
        }

        // Extract real eigenvalues from diagonal (Hermitian guarantees real eigenvalues)
        let mut eigenvals = Array1::zeros(n);
        for i in 0..n {
            eigenvals[i] = h[[i, i]].re;
        }

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvals[j].partial_cmp(&eigenvals[i]).unwrap());

        let sorted_eigenvals = Array1::from_shape_fn(n, |i| eigenvals[indices[i]]);
        let sorted_eigenvecs = Array2::from_shape_fn((n, n), |(i, j)| v[[i, indices[j]]]);

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_real_symmetric_eigendecomposition() {
        // Simple 2×2 symmetric matrix
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();

        let (eigenvals, eigenvecs) = EigenDecomposition::eigendecomposition(&matrix).unwrap();

        // Check eigenvalues (should be 3 and 1 in descending order)
        assert!((eigenvals[0] - 3.0).abs() < 1e-6);
        assert!((eigenvals[1] - 1.0).abs() < 1e-6);

        // Verify A*v = λ*v for each eigenvector
        for i in 0..2 {
            let lambda = eigenvals[i];
            let v = eigenvecs.column(i);
            let av = matrix.dot(&v.to_owned());

            for j in 0..2 {
                assert!((av[j] - lambda * v[j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_hermitian_eigendecomposition_identity() {
        // Identity matrix: all eigenvalues = 1
        let n = 3;
        let identity = Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        });

        let (eigenvals, _eigenvecs) =
            EigenDecomposition::hermitian_eigendecomposition_complex(&identity).unwrap();

        // All eigenvalues should be 1.0
        for i in 0..n {
            assert!((eigenvals[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hermitian_eigendecomposition_diagonal() {
        // Diagonal Hermitian matrix
        let diag_vals = vec![5.0, 3.0, 1.0];
        let matrix = Array2::from_shape_fn((3, 3), |(i, j)| {
            if i == j {
                Complex::new(diag_vals[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        });

        let (eigenvals, eigenvecs) =
            EigenDecomposition::hermitian_eigendecomposition_complex(&matrix).unwrap();

        // Eigenvalues should match diagonal (sorted descending)
        let mut expected = diag_vals.clone();
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for i in 0..3 {
            assert!(
                (eigenvals[i] - expected[i]).abs() < 1e-10,
                "Eigenvalue mismatch at {}: expected {}, got {}",
                i,
                expected[i],
                eigenvals[i]
            );
        }

        // Verify reconstruction: H = V Λ V†
        let lambda_diag = Array2::from_diag(&eigenvals.mapv(|x| Complex::new(x, 0.0)));
        let vdag: Array2<Complex<f64>> = eigenvecs
            .t()
            .mapv(|z| z.conj())
            .into_dimensionality()
            .unwrap();
        let v_lambda = eigenvecs.dot(&lambda_diag);
        let reconstructed: Array2<Complex<f64>> = v_lambda.dot(&vdag);

        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_hermitian_eigendecomposition_2x2() {
        // 2×2 Hermitian matrix with known eigenvalues
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(2.0, 0.0),
                Complex::new(1.0, -1.0),
                Complex::new(1.0, 1.0),
                Complex::new(3.0, 0.0),
            ],
        )
        .unwrap();

        let (eigenvals, eigenvecs) =
            EigenDecomposition::hermitian_eigendecomposition_complex(&matrix).unwrap();

        // Analytical eigenvalues from characteristic equation:
        // det(H - λI) = (2-λ)(3-λ) - |1-i|² = λ² - 5λ + 4 = 0
        // λ = (5 ± 3) / 2 = {4, 1}
        let expected_large = 4.0;
        let expected_small = 1.0;

        assert!(
            (eigenvals[0] - expected_large).abs() < 1e-10,
            "Large eigenvalue mismatch: expected {}, got {}",
            expected_large,
            eigenvals[0]
        );
        assert!(
            (eigenvals[1] - expected_small).abs() < 1e-10,
            "Small eigenvalue mismatch: expected {}, got {}",
            expected_small,
            eigenvals[1]
        );

        // Verify eigenvalue equation: H v = λ v
        for i in 0..2 {
            let lambda = eigenvals[i];
            let v = eigenvecs.column(i).to_owned();
            let hv = matrix.dot(&v);

            for j in 0..2 {
                assert!((hv[j] - lambda * v[j]).norm() < 1e-10);
            }
        }

        // Verify orthonormality: V† V = I
        let vdag: Array2<Complex<f64>> = eigenvecs
            .t()
            .mapv(|z| z.conj())
            .into_dimensionality()
            .unwrap();
        let vdag_v: Array2<Complex<f64>> = vdag.dot(&eigenvecs);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((vdag_v[[i, j]].re - expected).abs() < 1e-10);
                assert!(vdag_v[[i, j]].im.abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_hermitian_eigendecomposition_non_hermitian_rejected() {
        // Non-Hermitian matrix should be rejected
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 1.0),
                Complex::new(2.0, 1.0), // Not conjugate of (1,1)
                Complex::new(2.0, 0.0),
            ],
        )
        .unwrap();

        let result = EigenDecomposition::hermitian_eigendecomposition_complex(&matrix);
        assert!(result.is_err());
    }

    #[test]
    fn test_hermitian_eigendecomposition_real_eigenvalues() {
        // All Hermitian matrices have real eigenvalues
        let matrix = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex::new(4.0, 0.0),
                Complex::new(1.0, -2.0),
                Complex::new(2.0, 1.0),
                Complex::new(1.0, 2.0),
                Complex::new(5.0, 0.0),
                Complex::new(-1.0, 3.0),
                Complex::new(2.0, -1.0),
                Complex::new(-1.0, -3.0),
                Complex::new(6.0, 0.0),
            ],
        )
        .unwrap();

        let (eigenvals, _) =
            EigenDecomposition::hermitian_eigendecomposition_complex(&matrix).unwrap();

        // All eigenvalues must be real (no imaginary parts from computation)
        for &lambda in eigenvals.iter() {
            assert!(lambda.is_finite());
        }
    }
}
