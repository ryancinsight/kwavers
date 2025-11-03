//! Matrix utilities for beamforming algorithms
//!
//! Provides essential matrix operations including inversion and linear algebra
//! routines optimized for beamforming applications.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

/// Simple matrix inversion using Gauss-Jordan elimination
/// Returns None if matrix is singular
#[must_use]
pub fn invert_matrix(mat: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    let n = mat.nrows();
    if n != mat.ncols() {
        return None;
    }

    // Create augmented matrix [A | I]
    let mut aug = Array2::<Complex64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = mat[(i, j)];
        }
        aug[(i, n + i)] = Complex64::new(1.0, 0.0);
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut pivot_row = i;
        let mut max_val = aug[(i, i)].norm();
        for k in (i + 1)..n {
            let val = aug[(k, i)].norm();
            if val > max_val {
                max_val = val;
                pivot_row = k;
            }
        }

        // Check if matrix is singular
        if max_val < 1e-14 {
            return None;
        }

        // Swap rows if needed
        if pivot_row != i {
            for j in 0..(2 * n) {
                let temp = aug[(i, j)];
                aug[(i, j)] = aug[(pivot_row, j)];
                aug[(pivot_row, j)] = temp;
            }
        }

        // Scale pivot row
        let pivot = aug[(i, i)];
        for j in 0..(2 * n) {
            aug[(i, j)] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[(k, i)];
                // Store row i values to avoid borrow checker issues
                let row_i: Vec<Complex64> = (0..(2 * n)).map(|j| aug[(i, j)]).collect();
                for j in 0..(2 * n) {
                    aug[(k, j)] -= factor * row_i[j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[(i, j)] = aug[(i, n + j)];
        }
    }

    Some(inv)
}

/// Compute eigenvalues and eigenvectors of Hermitian matrix
/// Returns (eigenvalues, eigenvectors) where eigenvectors are columns
/// Uses power iteration for dominant eigenvalues
#[must_use]
pub fn eigen_hermitian(
    mat: &Array2<Complex64>,
    num_eigs: usize,
) -> Option<(Vec<f64>, Array2<Complex64>)> {
    let n = mat.nrows();
    if n != mat.ncols() || num_eigs == 0 || num_eigs > n {
        return None;
    }

    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Array2::<Complex64>::zeros((n, num_eigs));
    let mut a = mat.clone();

    for col in 0..num_eigs {
        // Power iteration for current eigenvalue
        let mut v = Array1::<Complex64>::from_vec(
            (0..n)
                .map(|i| Complex64::new((i + 1) as f64, 0.0))
                .collect(),
        );

        // Normalize
        let norm: f64 = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        v.mapv_inplace(|x| x / norm);

        for _ in 0..100 {
            // v = A * v
            let mut v_new = Array1::<Complex64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += a[(i, j)] * v[j];
                }
            }

            // Normalize
            let norm: f64 = v_new.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm < 1e-14 {
                break;
            }
            v_new.mapv_inplace(|x| x / norm);

            // Check convergence
            let diff: f64 = v
                .iter()
                .zip(v_new.iter())
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            v = v_new;

            if diff < 1e-10 {
                break;
            }
        }

        // Compute eigenvalue: λ = v^H A v
        let mut lambda = Complex64::zero();
        for i in 0..n {
            for j in 0..n {
                lambda += v[i].conj() * a[(i, j)] * v[j];
            }
        }

        eigenvalues.push(lambda.re);

        // Store eigenvector
        for i in 0..n {
            eigenvectors[(i, col)] = v[i];
        }

        // Deflate matrix: A = A - λ v v^H
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] -= lambda * v[i] * v[j].conj();
            }
        }
    }

    Some((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_matrix_inversion_identity() {
        let mut mat = Array2::<Complex64>::zeros((3, 3));
        mat[(0, 0)] = Complex64::new(1.0, 0.0);
        mat[(1, 1)] = Complex64::new(1.0, 0.0);
        mat[(2, 2)] = Complex64::new(1.0, 0.0);

        let inv = invert_matrix(&mat).unwrap();
        assert_relative_eq!(inv[(0, 0)].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(inv[(1, 1)].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(inv[(2, 2)].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_inversion_singular() {
        let mut mat = Array2::<Complex64>::zeros((2, 2));
        mat[(0, 0)] = Complex64::new(1.0, 0.0);
        mat[(0, 1)] = Complex64::new(2.0, 0.0);
        mat[(1, 0)] = Complex64::new(2.0, 0.0); // Linearly dependent rows
        mat[(1, 1)] = Complex64::new(4.0, 0.0);

        assert!(invert_matrix(&mat).is_none());
    }
}
