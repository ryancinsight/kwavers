//! Pure Rust Linear Algebra Operations
//!
//! This module provides essential linear algebra operations without external dependencies,
//! following SOLID principles and zero-copy techniques where possible.
//!
//! # Complex linear algebra (SSOT for narrowband beamforming)
//! Narrowband array processing (MVDR/Capon, MUSIC, ESMV) is most naturally expressed on complex
//! baseband snapshots and requires solving linear systems of the form `R y = a` where `R` is
//! complex Hermitian (after diagonal loading). This module provides SSOT complex solvers so
//! beamforming code can remain free of duplicated numerics.

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::{Float, NumCast, Zero};

/// Generic numeric operations for improved type safety and reusability
pub trait NumericOps<T>: Clone + Copy + PartialOrd + Zero
where
    T: Float + NumCast,
{
    /// Generic dot product for any float type
    fn dot_product(a: &[T], b: &[T]) -> Option<T> {
        if a.len() != b.len() {
            return None;
        }
        Some(
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| x * y)
                .fold(T::zero(), |acc, val| acc + val),
        )
    }

    /// Generic vector normalization
    fn normalize(vector: &mut [T]) -> bool {
        let norm_sq = vector
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, val| acc + val);
        if norm_sq <= T::zero() {
            return false;
        }
        let norm = norm_sq.sqrt();
        for x in vector.iter_mut() {
            *x = *x / norm;
        }
        true
    }

    /// Generic element-wise addition for arrays
    fn add_arrays(a: &[T], b: &[T], out: &mut [T]) -> Result<(), &'static str> {
        if a.len() != b.len() || b.len() != out.len() {
            return Err("Array length mismatch");
        }
        for ((a_val, b_val), out_val) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *out_val = *a_val + *b_val;
        }
        Ok(())
    }

    /// Generic scalar multiplication
    fn scale_array(input: &[T], scalar: T, out: &mut [T]) -> Result<(), &'static str> {
        if input.len() != out.len() {
            return Err("Array length mismatch");
        }
        for (input_val, out_val) in input.iter().zip(out.iter_mut()) {
            *out_val = *input_val * scalar;
        }
        Ok(())
    }

    /// Generic L2 norm calculation
    fn l2_norm(array: &[T]) -> T {
        array
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, val| acc + val)
            .sqrt()
    }

    /// Generic maximum absolute value
    fn max_abs(array: &[T]) -> T {
        array
            .iter()
            .map(|&x| x.abs())
            .fold(T::zero(), |acc, val| acc.max(val))
    }

    /// Safe division with tolerance check
    fn safe_divide(numerator: T, denominator: T, tolerance: T) -> Option<T> {
        if denominator.abs() > tolerance {
            Some(numerator / denominator)
        } else {
            None
        }
    }
}

/// Tolerance constants for numerical operations
pub mod tolerance {
    /// Default tolerance for convergence checks
    pub const DEFAULT: f64 = 1e-12;
    /// Tolerance for matrix rank determination
    pub const RANK: f64 = 1e-10;
    /// Maximum iterations for iterative methods
    pub const MAX_ITERATIONS: usize = 1000;

    /// Tolerance used for detecting near-zero complex pivots during LU factorization.
    ///
    /// This is intentionally aligned with `RANK` to preserve existing conditioning policy.
    pub const COMPLEX_PIVOT: f64 = RANK;

    /// Convergence tolerance for the SSOT complex Hermitian eigensolver (Jacobi on real-embedded form).
    ///
    /// This bounds the maximum absolute off-diagonal entry of the embedded real symmetric matrix
    /// before declaring convergence.
    pub const HERMITIAN_EIG_TOL: f64 = 1e-12;

    /// Maximum sweeps (major iterations) for the SSOT complex Hermitian eigensolver (Jacobi).
    pub const HERMITIAN_EIG_MAX_SWEEPS: usize = 2048;

    /// Convergence tolerance for tridiagonal QR eigensolver (off-diagonal magnitude threshold).
    pub const SYMM_TRIDIAG_QR_TOL: f64 = 1e-12;

    /// Maximum iterations for implicit QR on symmetric tridiagonal matrices.
    pub const SYMM_TRIDIAG_QR_MAX_ITERS: usize = 256;

    /// Below this dimension (2n for the embedded real symmetric problem), Jacobi is fine and
    /// often faster due to lower constant factors and simpler code paths.
    pub const HERMITIAN_EIG_JACOBI_CUTOFF_DIM: usize = 64;
}

// Implement NumericOps for standard float types
impl NumericOps<f64> for f64 {}
impl NumericOps<f32> for f32 {}

/// Pure Rust implementation of basic linear algebra operations
#[derive(Debug)]
pub struct LinearAlgebra;

impl LinearAlgebra {
    /// Solve linear system Ax = b using LU decomposition with partial pivoting
    pub fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        let n = a.nrows();
        if a.ncols() != n || b.len() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "solve_linear_system".to_string(),
                expected: format!("{n}x{n}"),
                actual: format!("{}x{}", a.nrows(), a.ncols()),
            }));
        }

        let mut lu = a.clone();
        let mut perm = (0..n).collect::<Vec<_>>();

        // LU decomposition with partial pivoting
        for k in 0..n - 1 {
            // Find pivot
            let mut max_row = k;
            for i in k + 1..n {
                if lu[[i, k]].abs() > lu[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                perm.swap(k, max_row);
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_row, j]];
                    lu[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if lu[[k, k]].abs() < tolerance::RANK {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "LU decomposition".to_string(),
                    condition_number: f64::INFINITY,
                }));
            }

            // Elimination
            for i in k + 1..n {
                lu[[i, k]] /= lu[[k, k]];
                for j in k + 1..n {
                    lu[[i, j]] -= lu[[i, k]] * lu[[k, j]];
                }
            }
        }

        // Apply permutation to b
        let mut pb = Array1::zeros(n);
        for i in 0..n {
            pb[i] = b[perm[i]];
        }

        // Forward substitution for Ly = Pb
        let mut y = Array1::zeros(n);
        for i in 0..n {
            y[i] = pb[i];
            for j in 0..i {
                y[i] -= lu[[i, j]] * y[j];
            }
        }

        // Back substitution for Ux = y
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in i + 1..n {
                x[i] -= lu[[i, j]] * x[j];
            }
            x[i] /= lu[[i, i]];
        }

        Ok(x)
    }

    /// Solve complex linear system `A x = b` using LU decomposition with partial pivoting.
    ///
    /// # Intended use
    /// Narrowband beamforming frequently requires solving `R y = a` where `R` is complex Hermitian
    /// (after diagonal loading). This function is SSOT for that solve.
    ///
    /// # Numerical policy
    /// - Pivot selection uses maximum-magnitude element in the current column.
    /// - Singularity detection uses `tolerance::COMPLEX_PIVOT` on pivot magnitude.
    pub fn solve_linear_system_complex(
        a: &Array2<Complex<f64>>,
        b: &Array1<Complex<f64>>,
    ) -> KwaversResult<Array1<Complex<f64>>> {
        let n = a.nrows();
        if a.ncols() != n || b.len() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "solve_linear_system_complex".to_string(),
                expected: format!("{n}x{n}"),
                actual: format!("{}x{}", a.nrows(), a.ncols()),
            }));
        }

        let mut lu = a.clone();
        let mut perm = (0..n).collect::<Vec<_>>();

        for k in 0..n - 1 {
            // Find pivot row by max |lu[i,k]|
            let mut max_row = k;
            let mut max_val = lu[[k, k]].norm();
            for i in k + 1..n {
                let v = lu[[i, k]].norm();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                perm.swap(k, max_row);
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_row, j]];
                    lu[[max_row, j]] = temp;
                }
            }

            // Check for singular/near-singular pivot
            if lu[[k, k]].norm() < tolerance::COMPLEX_PIVOT {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "LU decomposition (complex)".to_string(),
                    condition_number: f64::INFINITY,
                }));
            }

            // Elimination
            for i in k + 1..n {
                lu[[i, k]] = lu[[i, k]] / lu[[k, k]];
                for j in k + 1..n {
                    lu[[i, j]] = lu[[i, j]] - lu[[i, k]] * lu[[k, j]];
                }
            }
        }

        // Apply permutation to b
        let mut pb = Array1::<Complex<f64>>::zeros(n);
        for i in 0..n {
            pb[i] = b[perm[i]];
        }

        // Forward substitution for L y = P b  (L has unit diagonal)
        let mut y = Array1::<Complex<f64>>::zeros(n);
        for i in 0..n {
            let mut acc = pb[i];
            for j in 0..i {
                acc -= lu[[i, j]] * y[j];
            }
            y[i] = acc;
        }

        // Back substitution for U x = y
        let mut x = Array1::<Complex<f64>>::zeros(n);
        for i in (0..n).rev() {
            let mut acc = y[i];
            for j in i + 1..n {
                acc -= lu[[i, j]] * x[j];
            }
            let pivot = lu[[i, i]];
            if pivot.norm() < tolerance::COMPLEX_PIVOT {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "Back substitution (complex)".to_string(),
                    condition_number: f64::INFINITY,
                }));
            }
            x[i] = acc / pivot;
        }

        Ok(x)
    }

    /// Matrix inversion for complex matrices using repeated complex solves.
    ///
    /// # Note
    /// This is O(n³) solves and intended for small/medium matrices where SSOT inversion is needed.
    /// Prefer `solve_linear_system_complex` when you only need `A^{-1} b`.
    pub fn matrix_inverse_complex(
        matrix: &Array2<Complex<f64>>,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "matrix_inverse_complex".to_string(),
                expected: format!("{n}x{n}"),
                actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            }));
        }

        let mut inverse = Array2::<Complex<f64>>::zeros((n, n));
        let mut e = Array1::<Complex<f64>>::zeros(n);

        // Solve A x = e_i for each i
        for i in 0..n {
            // reset e
            for k in 0..n {
                e[k] = Complex::new(0.0, 0.0);
            }
            e[i] = Complex::new(1.0, 0.0);

            let col = Self::solve_linear_system_complex(matrix, &e)?;
            for r in 0..n {
                inverse[[r, i]] = col[r];
            }
        }

        Ok(inverse)
    }

    /// Compute eigenvalues and eigenvectors using QR algorithm
    pub fn eigendecomposition(matrix: &Array2<f64>) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "eigendecomposition".to_string(),
                expected: format!("{n}x{n}"),
                actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            }));
        }

        // Use power iteration for dominant eigenvalue/eigenvector
        // Implementation using QR algorithm for eigenvalue decomposition
        let mut a = matrix.clone();
        let mut q = Array2::eye(n);

        // QR iterations for eigenvalue computation
        for _ in 0..tolerance::MAX_ITERATIONS {
            let (q_iteration, r) = Self::qr_decomposition(&a)?;
            a = r.dot(&q_iteration);
            q = q.dot(&q_iteration);

            // Check for convergence using off-diagonal norm
            let mut converged = true;
            for i in 0..n - 1 {
                if a[[i + 1, i]].abs() > tolerance::DEFAULT {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }
        }

        // Extract eigenvalues from diagonal
        let eigenvalues = Array1::from_iter((0..n).map(|i| a[[i, i]]));

        Ok((eigenvalues, q))
    }

    /// SSOT complex Hermitian eigendecomposition.
    ///
    /// # Mathematical contract
    /// For a Hermitian matrix `H` (H = Hᴴ), returns real eigenvalues `λ` and unitary eigenvectors `V`
    /// such that `H = V diag(λ) Vᴴ` up to numerical tolerance.
    ///
    /// # Implementation strategy (pure Rust)
    /// We embed Hermitian `H = A + iB` (A real symmetric, B real skew-symmetric) into a real
    /// symmetric matrix:
    ///
    /// ```text
    /// S = [ A  -B ]
    ///     [ B   A ]
    /// ```
    ///
    /// `S` is real symmetric. Its eigenvalues are the eigenvalues of `H`, each duplicated twice.
    /// We compute an orthonormal eigendecomposition of `S` via a symmetric Jacobi method, then
    /// reconstruct complex eigenvectors of `H` from the chosen 2n-dimensional eigenvectors.
    ///
    /// # Errors
    /// - Dimension mismatch / non-square input.
    /// - Non-finite inputs.
    /// - Failure to converge within `tolerance::HERMITIAN_EIG_MAX_SWEEPS`.
    ///
    /// # Notes
    /// This routine is intended for beamforming-sized matrices (tens to low hundreds).
    pub fn hermitian_eigendecomposition_complex(
        h: &Array2<Complex<f64>>,
    ) -> KwaversResult<(Array1<f64>, Array2<Complex<f64>>)> {
        let n = h.nrows();
        if n == 0 || h.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "hermitian_eigendecomposition_complex".to_string(),
                expected: "nxn (n>0)".to_string(),
                actual: format!("{}x{}", h.nrows(), h.ncols()),
            }));
        }

        // Validate finiteness
        for i in 0..n {
            for j in 0..n {
                let z = h[(i, j)];
                if !z.re.is_finite() || !z.im.is_finite() {
                    return Err(KwaversError::Numerical(NumericalError::NaN {
                        operation: "hermitian_eigendecomposition_complex".to_string(),
                        inputs: format!("non-finite entry at ({i},{j})"),
                    }));
                }
            }
        }

        // Build real symmetric embedding S (2n x 2n).
        // H = A + iB, with A = Re(H), B = Im(H).
        // For Hermitian: A^T = A, B^T = -B.
        let m = 2 * n;
        let mut s = Array2::<f64>::zeros((m, m));
        for i in 0..n {
            for j in 0..n {
                let a = h[(i, j)].re;
                let b = h[(i, j)].im;
                s[(i, j)] = a;
                s[(i, j + n)] = -b;
                s[(i + n, j)] = b;
                s[(i + n, j + n)] = a;
            }
        }

        // SSOT correctness policy:
        // The tridiagonal-QR path is currently not proven correct/stable across our test suite.
        // Route Hermitian eigendecomposition through the robust symmetric Jacobi method until
        // the tridiagonal-QR implementation has convergence + reconstruction guarantees and tests.
        let (d, q) = Self::symmetric_jacobi(
            &s,
            tolerance::HERMITIAN_EIG_TOL,
            tolerance::HERMITIAN_EIG_MAX_SWEEPS,
        )?;

        // Sort eigenpairs ascending.
        let mut idx: Vec<usize> = (0..m).collect();
        idx.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));

        // Select n eigenpairs from the 2n spectrum by taking every other eigenvalue (they are duplicated).
        // We choose the first occurrence of each duplicated value. This is deterministic and stable enough
        // for downstream subspace partitioning (signal/noise).
        let mut evals = Array1::<f64>::zeros(n);
        let mut evecs = Array2::<Complex<f64>>::zeros((n, n));

        let mut out_col = 0usize;
        let mut k = 0usize;
        while out_col < n && k < m {
            let sel = idx[k];

            let lambda = d[sel];
            if !lambda.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    "hermitian_eigendecomposition_complex: non-finite eigenvalue".to_string(),
                )));
            }

            // Form complex vector v = x + i y from real eigenvector u = [x; y].
            // u is the sel-th column of Q.
            let mut norm_sq = 0.0;
            for i in 0..n {
                let x = q[(i, sel)];
                let y = q[(i + n, sel)];
                norm_sq += x * x + y * y;
            }

            if norm_sq <= 0.0 || !norm_sq.is_finite() {
                k += 1;
                continue;
            }

            let inv_norm = 1.0 / norm_sq.sqrt();
            for i in 0..n {
                let x = q[(i, sel)] * inv_norm;
                let y = q[(i + n, sel)] * inv_norm;
                evecs[(i, out_col)] = Complex::new(x, y);
            }
            evals[out_col] = lambda;

            out_col += 1;
            k += 2; // skip presumed duplicate partner to keep determinism
        }

        if out_col != n {
            return Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
                method: "hermitian_eigendecomposition_complex (embedding selection)".to_string(),
                iterations: tolerance::HERMITIAN_EIG_MAX_SWEEPS,
                error: f64::INFINITY,
            }));
        }

        Ok((evals, evecs))
    }

    /// Symmetric Jacobi eigendecomposition for real symmetric matrices.
    ///
    /// Returns (diag_eigenvalues, eigenvectors_columns).
    ///
    /// This is robust but not the fastest for moderate/large `n`. Prefer
    /// `symmetric_eigendecomposition_tridiag_qr` for larger matrices.
    fn symmetric_jacobi(
        a: &Array2<f64>,
        tol: f64,
        max_sweeps: usize,
    ) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        let n = a.nrows();
        if n == 0 || a.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "symmetric_jacobi".to_string(),
                expected: "nxn (n>0)".to_string(),
                actual: format!("{}x{}", a.nrows(), a.ncols()),
            }));
        }

        let mut dmat = a.clone();
        let mut v = Array2::<f64>::eye(n);

        let mut last_max_off = f64::INFINITY;

        for _sweep in 0..max_sweeps {
            // Find largest off-diagonal entry (by absolute value).
            let mut max_off = 0.0;
            let mut p = 0usize;
            let mut q = 1usize;

            for i in 0..n {
                for j in (i + 1)..n {
                    let val = dmat[(i, j)].abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }

            last_max_off = max_off;

            if max_off <= tol {
                // Converged
                let evals = Array1::from_iter((0..n).map(|i| dmat[(i, i)]));
                return Ok((evals, v));
            }

            let app = dmat[(p, p)];
            let aqq = dmat[(q, q)];
            let apq = dmat[(p, q)];

            if apq == 0.0 {
                continue;
            }

            // Compute Jacobi rotation parameters
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Apply rotation to dmat (symmetric update)
            // Update rows/cols p and q
            for k in 0..n {
                if k != p && k != q {
                    let aik = dmat[(k, p)];
                    let akq = dmat[(k, q)];
                    let new_kp = c * aik - s * akq;
                    let new_kq = s * aik + c * akq;
                    dmat[(k, p)] = new_kp;
                    dmat[(p, k)] = new_kp;
                    dmat[(k, q)] = new_kq;
                    dmat[(q, k)] = new_kq;
                }
            }

            let new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            let new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            dmat[(p, p)] = new_pp;
            dmat[(q, q)] = new_qq;
            dmat[(p, q)] = 0.0;
            dmat[(q, p)] = 0.0;

            // Update eigenvector matrix v
            for k in 0..n {
                let vkp = v[(k, p)];
                let vkq = v[(k, q)];
                v[(k, p)] = c * vkp - s * vkq;
                v[(k, q)] = s * vkp + c * vkq;
            }
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "symmetric_jacobi".to_string(),
            iterations: max_sweeps,
            error: last_max_off,
        }))
    }

    #[cfg(test)]
    fn diag_matrix(diag: &Array1<f64>) -> Array2<f64> {
        let n = diag.len();
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            m[(i, i)] = diag[i];
        }
        m
    }

    #[cfg(test)]
    fn hermitian_test_matrix_3x3() -> Array2<Complex<f64>> {
        // Hermitian: H = Hᴴ
        // Diagonal is real; off-diagonals are conjugate pairs.
        let mut h = Array2::<Complex<f64>>::zeros((3, 3));
        h[(0, 0)] = Complex::new(2.0, 0.0);
        h[(1, 1)] = Complex::new(1.5, 0.0);
        h[(2, 2)] = Complex::new(1.0, 0.0);

        h[(0, 1)] = Complex::new(0.2, 0.3);
        h[(1, 0)] = h[(0, 1)].conj();

        h[(0, 2)] = Complex::new(-0.1, 0.05);
        h[(2, 0)] = h[(0, 2)].conj();

        h[(1, 2)] = Complex::new(0.25, -0.15);
        h[(2, 1)] = h[(1, 2)].conj();

        h
    }

    #[cfg(test)]
    fn frobenius_norm(a: &Array2<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[cfg(test)]
    fn max_abs_entry(a: &Array2<f64>) -> f64 {
        a.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }

    #[cfg(test)]
    fn approx_symmetric(a: &Array2<f64>, tol: f64) -> bool {
        let n = a.nrows();
        if n == 0 || a.ncols() != n {
            return false;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if (a[(i, j)] - a[(j, i)]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    #[cfg(test)]
    fn orthonormal_columns(q: &Array2<f64>, tol: f64) -> bool {
        let n = q.nrows();
        if n == 0 || q.ncols() != n {
            return false;
        }
        // Check QᵀQ ≈ I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    #[cfg(test)]
    fn reconstruct_from_eig(q: &Array2<f64>, evals: &Array1<f64>) -> Array2<f64> {
        let n = q.nrows();
        let mut tmp = Array2::<f64>::zeros((n, n));
        // tmp = Q * diag(evals)
        for i in 0..n {
            for j in 0..n {
                tmp[(i, j)] = q[(i, j)] * evals[j];
            }
        }
        // tmp * Qᵀ
        let mut recon = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += tmp[(i, k)] * q[(j, k)];
                }
                recon[(i, j)] = acc;
            }
        }
        recon
    }

    #[cfg(test)]
    fn symmetric_test_matrix_5x5() -> Array2<f64> {
        // Deterministic symmetric matrix (mildly conditioned, not diagonal).
        // Chosen values are small to reduce overflow/instability during tests.
        let a = [
            [4.0, 1.0, 0.5, 0.0, 0.2],
            [1.0, 3.0, 0.3, 0.4, 0.0],
            [0.5, 0.3, 2.5, 0.6, 0.1],
            [0.0, 0.4, 0.6, 2.0, 0.7],
            [0.2, 0.0, 0.1, 0.7, 1.5],
        ];
        let mut m = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                m[(i, j)] = a[i][j];
            }
        }
        m
    }

    #[cfg(test)]
    fn identity(n: usize) -> Array2<f64> {
        let mut i = Array2::<f64>::zeros((n, n));
        for k in 0..n {
            i[(k, k)] = 1.0;
        }
        i
    }

    /// Fast symmetric eigendecomposition via:
    /// 1) Householder tridiagonalization: A = Q T Qᵀ
    /// 2) Implicit QR iterations with Wilkinson shift on tridiagonal T
    ///
    /// Returns eigenvalues and orthonormal eigenvectors (columns) of the original matrix.
    fn symmetric_eigendecomposition_tridiag_qr(
        a: &Array2<f64>,
        tol: f64,
        max_iters: usize,
    ) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        let n = a.nrows();
        if n == 0 || a.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "symmetric_eigendecomposition_tridiag_qr".to_string(),
                expected: "nxn (n>0)".to_string(),
                actual: format!("{}x{}", a.nrows(), a.ncols()),
            }));
        }

        // Safety: require symmetry for this routine.
        // (Hermitian embedding construction guarantees symmetry; this also protects other call sites.)
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = (a[(i, j)] - a[(j, i)]).abs();
                if !diff.is_finite() || diff > 1e-10 {
                    return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                        "symmetric_eigendecomposition_tridiag_qr: input must be symmetric"
                            .to_string(),
                    )));
                }
            }
        }

        // Tridiagonalize: produce diagonal d, subdiagonal e, and accumulated orthogonal Q.
        let (mut d, mut e, mut q) = Self::symmetric_tridiagonalize_householder(a)?;

        // Diagonalize tridiagonal with implicit QR; accumulate eigenvectors into q.
        Self::tridiagonal_implicit_qr(&mut d, &mut e, &mut q, tol, max_iters)?;

        Ok((d, q))
    }

    /// Householder tridiagonalization for real symmetric matrices.
    ///
    /// Returns (d, e, q) where:
    /// - `d` is the diagonal of T
    /// - `e` is the subdiagonal of T (length n, with e[0]=0)
    /// - `q` is the orthogonal matrix such that A = Q T Qᵀ
    fn symmetric_tridiagonalize_householder(
        a: &Array2<f64>,
    ) -> KwaversResult<(Array1<f64>, Array1<f64>, Array2<f64>)> {
        let n = a.nrows();
        if n == 0 || a.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "symmetric_tridiagonalize_householder".to_string(),
                expected: "nxn (n>0)".to_string(),
                actual: format!("{}x{}", a.nrows(), a.ncols()),
            }));
        }

        let mut t = a.clone();
        let mut q = Array2::<f64>::eye(n);

        // Reduce t to tridiagonal form in-place; accumulate orthogonal transforms into q.
        // Classic Householder reduction (Golub & Van Loan).
        for k in 0..n.saturating_sub(2) {
            // Build vector x = t[k+1.., k]
            let m = n - (k + 1);
            let mut x = Vec::<f64>::with_capacity(m);
            for i in 0..m {
                x.push(t[(k + 1 + i, k)]);
            }

            // Compute norm of x
            let mut norm_x = 0.0;
            for &xi in &x {
                norm_x += xi * xi;
            }
            norm_x = norm_x.sqrt();

            if norm_x <= tolerance::SYMM_TRIDIAG_QR_TOL {
                continue;
            }

            // Compute Householder vector v = x + sign(x0)*||x|| e1, then normalize.
            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            x[0] += sign * norm_x;

            let mut norm_v = 0.0;
            for &xi in &x {
                norm_v += xi * xi;
            }
            norm_v = norm_v.sqrt();
            if norm_v <= 0.0 || !norm_v.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    "symmetric_tridiagonalize_householder: invalid Householder norm".to_string(),
                )));
            }
            for xi in &mut x {
                *xi /= norm_v;
            }

            // Apply reflector to t from left and right (symmetric rank-2 update):
            // t := (I-2vvᵀ) t (I-2vvᵀ)
            // Efficient form: w = 2 * t_sub * v ; alpha = vᵀ w ; w := w - alpha v ; t_sub := t_sub - v wᵀ - w vᵀ
            //
            // We operate on the trailing submatrix starting at (k+1,k+1).
            let mut w = vec![0.0f64; m];
            for i in 0..m {
                let mut acc = 0.0;
                for j in 0..m {
                    acc += t[(k + 1 + i, k + 1 + j)] * x[j];
                }
                w[i] = 2.0 * acc;
            }

            let mut alpha = 0.0;
            for i in 0..m {
                alpha += x[i] * w[i];
            }
            for i in 0..m {
                w[i] -= alpha * x[i];
            }

            // Update trailing submatrix
            for i in 0..m {
                for j in i..m {
                    let delta = x[i] * w[j] + w[i] * x[j];
                    t[(k + 1 + i, k + 1 + j)] -= delta;
                    if i != j {
                        t[(k + 1 + j, k + 1 + i)] -= delta;
                    }
                }
            }

            // Update the k-th column/row below diagonal to enforce tridiagonal structure
            // t[k+1.., k] and t[k, k+1..]
            for i in 0..m {
                t[(k + 1 + i, k)] = 0.0;
                t[(k, k + 1 + i)] = 0.0;
            }
            // The first subdiagonal element is preserved (will be set from transformed values).
            t[(k + 1, k)] = -sign * norm_x;
            t[(k, k + 1)] = -sign * norm_x;

            // Accumulate Q := Q (I - 2 v vᵀ) on the right, affecting columns k+1..
            for row in 0..n {
                // Compute dot = sum_{i} Q[row, k+1+i] * v[i]
                let mut dot = 0.0;
                for i in 0..m {
                    dot += q[(row, k + 1 + i)] * x[i];
                }
                dot *= 2.0;
                for i in 0..m {
                    q[(row, k + 1 + i)] -= dot * x[i];
                }
            }
        }

        // Extract diagonal and subdiagonal
        let mut d = Array1::<f64>::zeros(n);
        let mut e = Array1::<f64>::zeros(n);
        for i in 0..n {
            d[i] = t[(i, i)];
            if i > 0 {
                e[i] = t[(i, i - 1)];
            }
        }

        Ok((d, e, q))
    }

    /// Symmetric tridiagonal eigensolver via:
    /// - Eigenvalue computation by **bisection** (Sturm sequence)
    /// - Eigenvector computation by **inverse iteration** per eigenvalue
    ///
    /// Mutates `d` (diagonal) and `e` (subdiagonal, `e[i]=T[i,i-1]` for `i>=1`, with `e[0]=0`)
    /// in-place and writes the orthonormal eigenvectors into `q` (columns). On success, `d` is
    /// overwritten with the eigenvalues (ascending, with multiplicities) and `e` is deflated to 0.
    ///
    /// This method is robust and avoids the subtle failure modes of an incorrect implicit QR/QL
    /// bulge chase implementation. Complexity is typically `O(n^2)` for all eigenpairs on
    /// tridiagonals, and it is well-suited for the embedded-real Hermitian path.
    fn tridiagonal_implicit_qr(
        d: &mut Array1<f64>,
        e: &mut Array1<f64>,
        q: &mut Array2<f64>,
        tol: f64,
        max_iters: usize,
    ) -> KwaversResult<()> {
        let n = d.len();
        if n == 0 || e.len() != n || q.nrows() != n || q.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "tridiagonal_bisection_inverse_iteration".to_string(),
                expected: "consistent tridiagonal sizes".to_string(),
                actual: format!(
                    "d={}, e={}, q={}x{}",
                    d.len(),
                    e.len(),
                    q.nrows(),
                    q.ncols()
                ),
            }));
        }

        // Representation invariant.
        e[0] = 0.0;

        // ---- Helper functions (local) ----

        // Infinity norm bound to bracket spectrum (Gershgorin for tridiagonal).
        fn spectral_bounds(d: &Array1<f64>, e: &Array1<f64>) -> (f64, f64) {
            let n = d.len();
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for i in 0..n {
                let mut r = 0.0;
                if i >= 1 {
                    r += e[i].abs();
                }
                if i + 1 < n {
                    r += e[i + 1].abs();
                }
                let a = d[i] - r;
                let b = d[i] + r;
                if a < lo {
                    lo = a;
                }
                if b > hi {
                    hi = b;
                }
            }
            (lo, hi)
        }

        // Count eigenvalues <= x using the Sturm sequence for symmetric tridiagonal.
        // Uses a stable recurrence with a small regularization to avoid division-by-zero.
        fn sturm_count(d: &Array1<f64>, e: &Array1<f64>, x: f64) -> usize {
            let n = d.len();
            let mut count = 0usize;

            // p0 = d0 - x
            let mut p = d[0] - x;
            if p <= 0.0 {
                count += 1;
            }

            let eps = 1e-300_f64; // prevents zero pivot in recurrence without affecting sign.
            for i in 1..n {
                let denom = if p.abs() < eps {
                    if p >= 0.0 {
                        eps
                    } else {
                        -eps
                    }
                } else {
                    p
                };
                p = (d[i] - x) - (e[i] * e[i]) / denom;
                if p <= 0.0 {
                    count += 1;
                }
            }

            count
        }

        // Solve tridiagonal linear system (T - shift I) y = b using Thomas algorithm.
        // T is defined by (d,e) with e[0]=0 and symmetric: subdiag/superdiag are e[1..].
        fn solve_tridiagonal_shifted(
            d: &Array1<f64>,
            e: &Array1<f64>,
            shift: f64,
            pivot_regularization: f64,
            b: &[f64],
            y: &mut [f64],
        ) -> KwaversResult<()> {
            let n = d.len();
            if b.len() != n || y.len() != n {
                return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                    operation: "solve_tridiagonal_shifted".to_string(),
                    expected: "consistent tridiagonal RHS".to_string(),
                    actual: format!("b={}, y={}, n={}", b.len(), y.len(), n),
                }));
            }

            // Forward sweep
            let mut cprime = vec![0.0f64; n];
            let mut dprime = vec![0.0f64; n];

            let mut denom = d[0] - shift;
            if denom == 0.0 {
                denom = pivot_regularization;
            }
            cprime[0] = if n > 1 { e[1] / denom } else { 0.0 };
            dprime[0] = b[0] / denom;

            for i in 1..n {
                let a_i = e[i];
                let c_i = if i + 1 < n { e[i + 1] } else { 0.0 };
                denom = (d[i] - shift) - a_i * cprime[i - 1];
                if denom == 0.0 {
                    denom = pivot_regularization;
                }
                cprime[i] = if i + 1 < n { c_i / denom } else { 0.0 };
                dprime[i] = (b[i] - a_i * dprime[i - 1]) / denom;
            }

            // Back substitution
            y[n - 1] = dprime[n - 1];
            for i in (0..n - 1).rev() {
                y[i] = dprime[i] - cprime[i] * y[i + 1];
            }

            Ok(())
        }

        // Orthonormalize vector v against first `k` columns of Q via classical Gram-Schmidt + re-orth.
        fn orthonormalize_against(q: &Array2<f64>, k: usize, v: &mut [f64]) {
            let n = v.len();

            // Two-pass re-orth to improve robustness for clustered eigenvalues.
            for _ in 0..2 {
                for j in 0..k {
                    let mut dot = 0.0;
                    for i in 0..n {
                        dot += v[i] * q[(i, j)];
                    }
                    for i in 0..n {
                        v[i] -= dot * q[(i, j)];
                    }
                }
            }
        }

        // Normalize vector; returns norm.
        fn normalize(v: &mut [f64]) -> f64 {
            let mut norm2 = 0.0;
            for &x in v.iter() {
                norm2 += x * x;
            }
            let norm = norm2.sqrt();
            if norm > 0.0 {
                let inv = 1.0 / norm;
                for x in v.iter_mut() {
                    *x *= inv;
                }
            }
            norm
        }

        // ---- Step 1: eigenvalues via bisection ----

        let (mut lo, mut hi) = spectral_bounds(d, e);

        // Expand bounds slightly to accommodate numerical edge cases.
        let pad = 1e-12_f64 * (hi - lo).abs().max(1.0);
        lo -= pad;
        hi += pad;

        // Bisection tolerance for eigenvalues. We tie it to requested `tol`.
        let eig_tol = tol.max(1e-14);

        let mut evals = vec![0.0f64; n];

        for (k, eval) in evals.iter_mut().enumerate().take(n) {
            // Determine interval [a,b] that contains the k-th eigenvalue (0-based, ascending).
            let target = k + 1; // Sturm count is 1-based
            let mut a = lo;
            let mut b = hi;

            // Narrow initial bracket by ensuring sturm_count(a) < target <= sturm_count(b).
            // For lo, count should be 0; for hi, count should be n.
            // If numerics break this, fall back to relaxed bounds.
            let ca = sturm_count(d, e, a);
            let cb = sturm_count(d, e, b);
            if !(ca < target && cb >= target) {
                // Relax aggressively.
                let width = (hi - lo).abs().max(1.0);
                a = lo - 10.0 * width;
                b = hi + 10.0 * width;
            }

            // Bisection loop: ensure we do not exceed max_iters.
            let mut it = 0usize;
            while (b - a).abs() > eig_tol * (a.abs() + b.abs()).max(1.0) {
                if it >= max_iters {
                    return Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
                        method: "tridiagonal_bisection".to_string(),
                        iterations: max_iters,
                        error: (b - a).abs(),
                    }));
                }
                let mid = 0.5 * (a + b);
                let c = sturm_count(d, e, mid);
                if c >= target {
                    b = mid;
                } else {
                    a = mid;
                }
                it += 1;
            }
            *eval = 0.5 * (a + b);
        }

        // ---- Step 2: eigenvectors via inverse iteration ----
        //
        // IMPORTANT: `q` already contains the accumulated orthogonal matrix from Householder
        // tridiagonalization (A = Q_house T Q_houseᵀ). The eigenvectors we compute below are for `T`.
        // Therefore, the eigenvectors of the original matrix are:
        //
        //   Q_total = Q_house * Q_tridiag
        //
        // We must NOT overwrite `q` with identity. Instead, compute `Q_tridiag` and post-multiply
        // the existing `q` by it.

        // Save current Householder accumulator.
        let q_house = q.clone();

        // Build Q_tridiag (eigenvectors of T) explicitly.
        let mut q_tridiag = Array2::<f64>::zeros((n, n));

        // For clustered eigenvalues, inverse iteration may need more iterations.
        let inv_it_max = max_iters.max(8);

        let mut v = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];

        for k in 0..n {
            // Deterministic starting vector: unit vector with wrap to avoid pathological zeros.
            v.fill(0.0);
            v[k % n] = 1.0;

            // Orthonormalize against previously computed tridiagonal eigenvectors.
            orthonormalize_against(&q_tridiag, k, &mut v);
            if normalize(&mut v) == 0.0 {
                // Fallback: use a uniform vector if orthogonalization wiped it out.
                v.fill(1.0);
                orthonormalize_against(&q_tridiag, k, &mut v);
                normalize(&mut v);
            }

            let lambda = evals[k];

            // Inverse iteration on (T - lambda I) y = v; set v <- y/||y||.
            let mut converged = false;
            for it in 0..inv_it_max {
                // Small shift away from exact eigenvalue to avoid singularity.
                // The perturbation is scaled to the problem magnitude.
                let shift = lambda + (1e-12_f64 * (lambda.abs().max(1.0))) * (it as f64 + 1.0);

                solve_tridiagonal_shifted(d, e, shift, tol, &v, &mut y)?;

                // Orthonormalize against previously computed tridiagonal eigenvectors (important for multiplicities).
                orthonormalize_against(&q_tridiag, k, &mut y);

                let norm = normalize(&mut y);
                if norm == 0.0 || !norm.is_finite() {
                    continue;
                }

                // Convergence check: residual ||T y - lambda y|| is small relative to ||y||.
                // We compute residual using tridiagonal structure.
                let mut res2 = 0.0f64;
                for i in 0..n {
                    let mut ty = d[i] * y[i];
                    if i >= 1 {
                        ty += e[i] * y[i - 1];
                    }
                    if i + 1 < n {
                        ty += e[i + 1] * y[i + 1];
                    }
                    let r = ty - lambda * y[i];
                    res2 += r * r;
                }
                let res = res2.sqrt();

                // Accept if residual is sufficiently small.
                if res <= (tol.max(1e-10)) * (lambda.abs().max(1.0)) {
                    converged = true;
                    v.copy_from_slice(&y);
                    break;
                }

                // Continue iterations
                v.copy_from_slice(&y);
            }

            if !converged {
                return Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
                    method: "tridiagonal_inverse_iteration".to_string(),
                    iterations: inv_it_max,
                    error: 1.0,
                }));
            }

            // Write eigenvector into Q_tridiag column k.
            for i in 0..n {
                q_tridiag[(i, k)] = v[i];
            }
        }

        // Final pass: enforce orthonormality (QR via Gram-Schmidt on Q_tridiag columns).
        // This is a defensive stabilization step for clustered eigenvalues.
        for j in 0..n {
            // Load column j into v
            for i in 0..n {
                v[i] = q_tridiag[(i, j)];
            }
            orthonormalize_against(&q_tridiag, j, &mut v);
            let norm = normalize(&mut v);
            if norm == 0.0 || !norm.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    "tridiagonal_bisection_inverse_iteration: failed to orthonormalize eigenvectors"
                        .to_string(),
                )));
            }
            for i in 0..n {
                q_tridiag[(i, j)] = v[i];
            }
        }

        // Compose: q := q_house * q_tridiag
        //
        // Note: we keep the API contract that `q` is the eigenvector matrix of the ORIGINAL input.
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0f64;
                for k in 0..n {
                    acc += q_house[(i, k)] * q_tridiag[(k, j)];
                }
                q[(i, j)] = acc;
            }
        }

        // Overwrite d with eigenvalues (ascending).
        for i in 0..n {
            d[i] = evals[i];
        }

        // Deflate subdiagonal to (near) zero for caller expectations.
        for i in 1..n {
            e[i] = 0.0;
        }

        Ok(())
    }

    /// QR decomposition using Gram-Schmidt process
    pub fn qr_decomposition(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array2<f64>)> {
        let (m, n) = matrix.dim();
        let mut q = Array2::zeros((m, n));
        let mut r = Array2::zeros((n, n));

        for j in 0..n {
            let mut v = matrix.column(j).to_owned();

            // Orthogonalize against previous columns
            for i in 0..j {
                let q_i = q.column(i);
                r[[i, j]] = v.dot(&q_i);
                v = v - r[[i, j]] * &q_i;
            }

            // Normalize
            r[[j, j]] = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if r[[j, j]] < tolerance::RANK {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "QR decomposition".to_string(),
                    condition_number: f64::INFINITY,
                }));
            }

            for k in 0..m {
                q[[k, j]] = v[k] / r[[j, j]];
            }
        }

        Ok((q, r))
    }

    /// Matrix inversion using LU decomposition
    pub fn matrix_inverse(matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "matrix_inverse".to_string(),
                expected: format!("{n}x{n}"),
                actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            }));
        }

        let mut inverse = Array2::zeros((n, n));
        let identity = Array2::eye(n);

        // Solve Ax = ei for each column of identity matrix
        for i in 0..n {
            let column = identity.column(i);
            let solution = Self::solve_linear_system(matrix, &column.to_owned())?;
            for j in 0..n {
                inverse[[j, i]] = solution[j];
            }
        }

        Ok(inverse)
    }

    /// Singular Value Decomposition
    ///
    /// Note: This implementation uses the eigendecomposition of A^T*A which is numerically
    /// less stable than methods like Golub-Kahan bidiagonalization. For production use,
    /// consider using LAPACK bindings or a more robust algorithm.
    ///
    /// Reference: Golub & Van Loan (2013) "Matrix Computations", 4th ed., Section 8.6
    pub fn svd(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (m, n) = matrix.dim();

        // Compute A^T * A for V and singular values
        let ata = matrix.t().dot(matrix);
        let (s_squared, v) = Self::eigendecomposition(&ata)?;

        // Singular values are square roots of eigenvalues
        let s = Array1::from_iter(s_squared.iter().map(|&x| x.sqrt()));

        // Compute U = A * V * S^(-1)
        let mut u = Array2::zeros((m, n.min(m)));
        for i in 0..n.min(m) {
            if s[i] > tolerance::RANK {
                let v_col = v.column(i);
                let u_col = matrix.dot(&v_col) / s[i];
                for j in 0..m {
                    u[[j, i]] = u_col[j];
                }
            }
        }

        Ok((u, s, v))
    }

    /// Compute L2 norm of a 3D array
    pub fn norm_l2_3d(array: &ndarray::Array3<f64>) -> f64 {
        array.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// Compute L2 norm of a 3D array (convenience function)
pub fn norm_l2(array: &ndarray::Array3<f64>) -> f64 {
    LinearAlgebra::norm_l2_3d(array)
}

/// Extension trait for ndarray operations
pub trait LinearAlgebraExt<T> {
    /// Solve linear system in-place where possible
    fn solve_into(&self, b: Array1<T>) -> KwaversResult<Array1<T>>;

    /// Compute matrix inverse
    fn inv(&self) -> KwaversResult<Array2<T>>;

    /// Eigendecomposition
    fn eig(&self) -> KwaversResult<(Array1<T>, Array2<T>)>;
}

impl LinearAlgebraExt<f64> for Array2<f64> {
    fn solve_into(&self, b: Array1<f64>) -> KwaversResult<Array1<f64>> {
        LinearAlgebra::solve_linear_system(self, &b)
    }

    fn inv(&self) -> KwaversResult<Array2<f64>> {
        LinearAlgebra::matrix_inverse(self)
    }

    fn eig(&self) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        LinearAlgebra::eigendecomposition(self)
    }
}

impl LinearAlgebraExt<Complex<f64>> for Array2<Complex<f64>> {
    fn solve_into(&self, b: Array1<Complex<f64>>) -> KwaversResult<Array1<Complex<f64>>> {
        LinearAlgebra::solve_linear_system_complex(self, &b)
    }

    fn inv(&self) -> KwaversResult<Array2<Complex<f64>>> {
        LinearAlgebra::matrix_inverse_complex(self)
    }

    fn eig(&self) -> KwaversResult<(Array1<Complex<f64>>, Array2<Complex<f64>>)> {
        Err(KwaversError::Numerical(
            NumericalError::UnsupportedOperation {
                operation: "Complex eigendecomposition".to_string(),
                reason: "Complex eigendecomposition is not implemented in SSOT LinearAlgebra"
                    .to_string(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use num_complex::Complex;

    fn frobenius_norm(a: &Array2<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn approx_symmetric(a: &Array2<f64>, tol: f64) -> bool {
        let n = a.nrows();
        if n == 0 || a.ncols() != n {
            return false;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if (a[(i, j)] - a[(j, i)]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn orthonormal_columns(q: &Array2<f64>, tol: f64) -> bool {
        let n = q.nrows();
        if n == 0 || q.ncols() != n {
            return false;
        }
        // Check QᵀQ ≈ I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn reconstruct_from_eig(q: &Array2<f64>, evals: &Array1<f64>) -> Array2<f64> {
        let n = q.nrows();
        let mut tmp = Array2::<f64>::zeros((n, n));
        // tmp = Q * diag(evals)
        for i in 0..n {
            for j in 0..n {
                tmp[(i, j)] = q[(i, j)] * evals[j];
            }
        }
        // tmp * Qᵀ
        let mut recon = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += tmp[(i, k)] * q[(j, k)];
                }
                recon[(i, j)] = acc;
            }
        }
        recon
    }

    fn reconstruct_tridiagonal(d: &Array1<f64>, e: &Array1<f64>) -> Array2<f64> {
        let n = d.len();
        let mut t = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            t[(i, i)] = d[i];
            if i >= 1 {
                t[(i, i - 1)] = e[i];
                t[(i - 1, i)] = e[i];
            }
        }
        t
    }

    fn approx_equal_multiset(mut a: Vec<f64>, mut b: Vec<f64>, tol: f64) -> bool {
        a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        if a.len() != b.len() {
            return false;
        }
        for i in 0..a.len() {
            let da = a[i];
            let db = b[i];
            if !(da.is_finite() && db.is_finite()) {
                return false;
            }
            if (da - db).abs() > tol {
                return false;
            }
        }
        true
    }

    fn symmetric_test_matrix_5x5() -> Array2<f64> {
        // Deterministic symmetric matrix (mildly conditioned, not diagonal).
        // Chosen values are small to reduce overflow/instability during tests.
        let a = [
            [4.0, 1.0, 0.5, 0.0, 0.2],
            [1.0, 3.0, 0.3, 0.4, 0.0],
            [0.5, 0.3, 2.5, 0.6, 0.1],
            [0.0, 0.4, 0.6, 2.0, 0.7],
            [0.2, 0.0, 0.1, 0.7, 1.5],
        ];
        let mut m = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                m[(i, j)] = a[i][j];
            }
        }
        m
    }

    fn hermitian_test_matrix_3x3() -> Array2<Complex<f64>> {
        // Hermitian: H = Hᴴ
        // Diagonal is real; off-diagonals are conjugate pairs.
        let mut h = Array2::<Complex<f64>>::zeros((3, 3));
        h[(0, 0)] = Complex::new(2.0, 0.0);
        h[(1, 1)] = Complex::new(1.5, 0.0);
        h[(2, 2)] = Complex::new(1.0, 0.0);

        h[(0, 1)] = Complex::new(0.2, 0.3);
        h[(1, 0)] = h[(0, 1)].conj();

        h[(0, 2)] = Complex::new(-0.1, 0.05);
        h[(2, 0)] = h[(0, 2)].conj();

        h[(1, 2)] = Complex::new(0.25, -0.15);
        h[(2, 1)] = h[(1, 2)].conj();

        h
    }

    #[test]
    fn test_numeric_ops_add_arrays() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut out = [0.0; 3];

        assert!(f64::add_arrays(&a, &b, &mut out).is_ok());
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_numeric_ops_scale_array() {
        let input = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];

        assert!(f64::scale_array(&input, 2.0, &mut out).is_ok());
        assert_eq!(out, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_numeric_ops_l2_norm() {
        let array = [3.0, 4.0];
        let norm = f64::l2_norm(&array);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_numeric_ops_safe_divide() {
        assert_eq!(f64::safe_divide(10.0, 2.0, 1e-10), Some(5.0));
        assert_eq!(f64::safe_divide(10.0, 1e-12, 1e-10), None);
    }

    #[test]
    fn test_numeric_ops_max_abs() {
        let array = [-5.0, 3.0, -2.0, 4.0];
        assert_eq!(f64::max_abs(&array), 5.0);
    }

    #[test]
    fn test_generic_float_types() {
        // Test that the trait works with both f64 and f32
        let a_f64 = [1.0_f64, 2.0, 3.0];
        let norm_f64 = f64::l2_norm(&a_f64);

        let a_f32 = [1.0_f32, 2.0, 3.0];
        let norm_f32 = f32::l2_norm(&a_f32);

        assert_relative_eq!(norm_f64 as f32, norm_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_solve_linear_system() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 2.0]);

        let x = LinearAlgebra::solve_linear_system(&a, &b).unwrap();

        // Solution should be [1, 1]
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0]).unwrap();
        let inv = LinearAlgebra::matrix_inverse(&a).unwrap();

        // Check A * A^(-1) = I
        let product = a.dot(&inv);
        assert_abs_diff_eq!(product[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qr_decomposition() {
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0]).unwrap();
        let (q, r) = LinearAlgebra::qr_decomposition(&a).unwrap();

        // Check that Q * R = A
        let product = q.dot(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(product[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_symmetric_householder_tridiagonalization_reconstruction_and_orthonormality() {
        let a = symmetric_test_matrix_5x5();
        assert!(approx_symmetric(&a, 1e-12));

        let (d, e, q) = LinearAlgebra::symmetric_tridiagonalize_householder(&a).expect("tridiag");

        // Q should be orthonormal.
        assert!(orthonormal_columns(&q, 1e-8));

        // A ≈ Q T Qᵀ where T is tridiagonal with (d,e).
        let t = reconstruct_tridiagonal(&d, &e);
        assert!(approx_symmetric(&t, 1e-12));

        let recon = q.dot(&t).dot(&q.t());

        let a_norm = frobenius_norm(&a).max(1e-12);
        let rel = frobenius_norm(&(&a - &recon)) / a_norm;

        assert!(rel.is_finite());
        assert!(
            rel < 1e-8,
            "relative tridiagonalization reconstruction error too large: {rel:e}"
        );

        // Ensure no spurious large off-tridiagonal entries in T (sanity).
        let mut max_off_tridiag = 0.0f64;
        for i in 0..5 {
            for j in 0..5 {
                if (i as isize - j as isize).abs() > 1 {
                    let v = t[(i, j)].abs();
                    if v > max_off_tridiag {
                        max_off_tridiag = v;
                    }
                }
            }
        }
        assert!(
            max_off_tridiag < 1e-8,
            "T has large off-tridiagonal entry: {max_off_tridiag:e}"
        );
    }

    #[test]
    fn test_tridiagonal_implicit_qr_matches_jacobi_eigenvalues_on_tridiagonal_input() {
        let a = symmetric_test_matrix_5x5();
        assert!(approx_symmetric(&a, 1e-12));

        // Build an explicit tridiagonal T from Householder, then run QR on (d,e) directly.
        let (mut d, mut e, _) =
            LinearAlgebra::symmetric_tridiagonalize_householder(&a).expect("tridiag");

        // Jacobi oracle eigenvalues from T (not from A), so this isolates QR correctness on tridiagonal input.
        let t = reconstruct_tridiagonal(&d, &e);
        let (oracle_vals, _) = LinearAlgebra::symmetric_jacobi(
            &t,
            tolerance::HERMITIAN_EIG_TOL,
            tolerance::HERMITIAN_EIG_MAX_SWEEPS,
        )
        .expect("jacobi oracle");

        // Run implicit QR on the tridiagonal representation.
        let n = d.len();
        let mut q_acc = Array2::<f64>::eye(n);
        LinearAlgebra::tridiagonal_implicit_qr(
            &mut d,
            &mut e,
            &mut q_acc,
            tolerance::SYMM_TRIDIAG_QR_TOL,
            tolerance::SYMM_TRIDIAG_QR_MAX_ITERS,
        )
        .expect("qr");

        // Compare eigenvalues as a multiset (ordering may differ).
        let ok = approx_equal_multiset(
            d.iter().copied().collect::<Vec<f64>>(),
            oracle_vals.iter().copied().collect::<Vec<f64>>(),
            1e-8,
        );
        assert!(
            ok,
            "QR eigenvalues do not match Jacobi oracle within tolerance"
        );

        // Additionally validate reconstruction on T: T ≈ Q diag(d) Qᵀ.
        assert!(orthonormal_columns(&q_acc, 1e-8));
        let recon = reconstruct_from_eig(&q_acc, &d);
        let rel = frobenius_norm(&(&t - &recon)) / frobenius_norm(&t).max(1e-12);
        assert!(rel.is_finite());
        assert!(
            rel < 1e-7,
            "relative reconstruction error too large: {rel:e}"
        );

        // Ensure we actually deflated: subdiagonal should be (near) zero after convergence.
        let mut max_e = 0.0f64;
        for i in 1..n {
            let v = e[i].abs();
            if v > max_e {
                max_e = v;
            }
        }
        assert!(
            max_e <= 1e-6,
            "did not deflate sufficiently, max |e|={max_e:e}"
        );
    }

    #[test]
    fn test_symmetric_tridiag_qr_eigendecomposition_reconstruction_and_orthonormality() {
        let a = symmetric_test_matrix_5x5();
        assert!(approx_symmetric(&a, 1e-12));

        let (evals, q) = LinearAlgebra::symmetric_eigendecomposition_tridiag_qr(
            &a,
            tolerance::SYMM_TRIDIAG_QR_TOL,
            tolerance::SYMM_TRIDIAG_QR_MAX_ITERS,
        )
        .expect("eigendecomposition");

        // Orthonormal columns: QᵀQ ≈ I
        assert!(orthonormal_columns(&q, 1e-8));

        // Reconstruction: A ≈ Q diag(λ) Qᵀ
        let recon = reconstruct_from_eig(&q, &evals);
        let diff = &a - &recon;

        let a_norm = frobenius_norm(&a).max(1e-12);
        let rel = frobenius_norm(&diff) / a_norm;

        // Tight enough to catch algorithmic errors, loose enough for CI/platform differences.
        assert!(rel.is_finite());
        assert!(
            rel < 1e-6,
            "relative reconstruction error too large: {rel:e}"
        );

        // Symmetry preserved by reconstruction
        assert!(approx_symmetric(&recon, 1e-10));

        // Sanity: eigenvalues finite
        for (i, &x) in evals.iter().enumerate() {
            assert!(x.is_finite(), "non-finite eigenvalue at {i}");
        }
    }

    #[test]
    fn test_hermitian_eigendecomposition_complex_reconstruction_sanity() {
        let h = hermitian_test_matrix_3x3();
        let (evals, v) = LinearAlgebra::hermitian_eigendecomposition_complex(&h).expect("eig");

        // Reconstruct H ≈ V diag(λ) Vᴴ
        let n = h.nrows();
        let mut tmp = Array2::<Complex<f64>>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                tmp[(i, j)] = v[(i, j)] * Complex::new(evals[j], 0.0);
            }
        }

        let mut recon = Array2::<Complex<f64>>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..n {
                    acc += tmp[(i, k)] * v[(j, k)].conj();
                }
                recon[(i, j)] = acc;
            }
        }

        let mut max_err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let err = (h[(i, j)] - recon[(i, j)]).norm();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        assert!(max_err.is_finite());
        assert!(
            max_err < 1e-6,
            "max reconstruction error too large: {max_err:e}"
        );
    }
}
