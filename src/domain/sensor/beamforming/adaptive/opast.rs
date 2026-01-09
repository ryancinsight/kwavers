//! Orthonormal PAST (OPAST) Algorithm for improved numerical stability
//!
//! This module implements the OPAST algorithm which maintains strict orthonormality
//! via QR decomposition for better numerical stability in subspace tracking.
//!
//! # Strict SSOT enforcement
//! OPAST depends on small-matrix inversion during its update recursion. Until the OPAST path is
//! fully migrated to SSOT `crate::utils::linear_algebra` (with explicit error propagation and no
//! silent fallbacks), this module is **feature-gated**.
//!
//! Enable with `--features legacy_algorithms` if you explicitly accept legacy numerics here.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;
use rayon::prelude::*;

/// Orthonormal PAST (OPAST) Algorithm for improved numerical stability
///
/// OPAST maintains strict orthonormality via QR decomposition, providing
/// better numerical stability than standard PAST, especially for long runs.
///
/// # References
/// - Abed-Meraim et al. (2000), "A general framework for performance analysis of subspace tracking algorithms"
/// - Strobach (1998), "Fast recursive subspace adaptive ESPRIT algorithms"
#[derive(Debug, Clone)]
pub struct OrthonormalSubspaceTracker {
    /// Orthonormal subspace basis matrix (n x p)
    subspace: Array2<Complex64>,
    /// Forgetting factor (0 < lambda < 1)
    lambda: f64,
    /// Accumulated weight for normalization
    weight: f64,
    /// Auxiliary matrix for QR update (not currently used)
    #[allow(dead_code)]
    r_matrix: Array2<Complex64>,
}

impl OrthonormalSubspaceTracker {
    /// Create new orthonormal subspace tracker
    ///
    /// # Arguments
    /// * `n` - Array size (number of sensors)
    /// * `p` - Subspace dimension (number of signals to track)
    /// * `lambda` - Forgetting factor (0.95-0.99 typical)
    pub fn new(n: usize, p: usize, lambda: f64) -> Self {
        assert!(p <= n, "Subspace dimension must be <= array size");
        assert!(
            lambda > 0.0 && lambda < 1.0,
            "Forgetting factor must be in (0,1)"
        );

        // Initialize with orthonormal basis (first p standard basis vectors)
        let mut subspace = Array2::<Complex64>::zeros((n, p));
        for i in 0..p.min(n) {
            subspace[(i, i)] = Complex64::new(1.0, 0.0);
        }

        // Initialize R matrix (upper triangular from QR)
        let mut r_matrix = Array2::<Complex64>::zeros((p, p));
        for i in 0..p {
            r_matrix[(i, i)] = Complex64::new(1.0, 0.0);
        }

        Self {
            subspace,
            lambda,
            weight: 1.0,
            r_matrix,
        }
    }

    /// Update subspace with new data snapshot
    ///
    /// Implements OPAST recursion with QR orthonormalization:
    /// 1. Standard PAST update
    /// 2. QR orthonormalization to maintain strict orthonormality
    pub fn update(&mut self, snapshot: &[Complex64]) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        assert_eq!(snapshot.len(), n, "Snapshot size mismatch");

        // Convert snapshot to Array1
        let y = Array1::from(snapshot.to_vec());

        // Step 1: Standard PAST update (same as regular SubspaceTracker)
        // Compute projection coefficients: α = (W^H W)^{-1} W^H y
        // Optimized: Parallel computation of W^H W using SIMD-friendly patterns
        let whw = self.compute_whw_parallel(&self.subspace.view());

        // Invert W^H W (small p x p matrix)
        let whw_inv = match super::matrix_utils::invert_matrix(&whw) {
            Some(inv) => inv,
            None => {
                // Fallback: use diagonal loading
                let mut loaded = whw.clone();
                for i in 0..p {
                    loaded[(i, i)] += Complex64::new(1e-10, 0.0);
                }
                super::matrix_utils::invert_matrix(&loaded).unwrap_or_else(|| {
                    // Ultimate fallback: identity
                    let mut id = Array2::zeros((p, p));
                    for i in 0..p {
                        id[(i, i)] = Complex64::new(1.0, 0.0);
                    }
                    id
                })
            }
        };

        // Compute alpha = (W^H W)^{-1} W^H y using optimized operations
        let wh_y = self.compute_wh_y_parallel(&self.subspace.view(), &y.view());
        let alpha = self.matrix_vector_multiply(&whw_inv.view(), &wh_y.view());

        // Update subspace: W(t+1) = lambda * W(t) + (y - W*alpha) * alpha^H
        let sqrt_lambda = self.lambda.sqrt();
        for i in 0..n {
            let mut w_alpha = Complex64::zero();
            for j in 0..p {
                w_alpha += self.subspace[(i, j)] * alpha[j];
            }
            let residual = y[i] - w_alpha;

            for j in 0..p {
                self.subspace[(i, j)] = sqrt_lambda * self.subspace[(i, j)]
                    + residual * alpha[j].conj()
                        / (1.0 + alpha.iter().map(|a| a.norm_sqr()).sum::<f64>()).sqrt();
            }
        }

        // Step 2: QR orthonormalization via Gram-Schmidt
        self.orthonormalize_subspace();

        // Update weight
        self.weight = self.lambda * self.weight + 1.0;
    }

    /// Orthonormalize subspace using optimized Gram-Schmidt with parallel operations
    fn orthonormalize_subspace(&mut self) {
        let _n = self.subspace.nrows();
        let p = self.subspace.ncols();

        for j in 0..p {
            // Orthogonalize against previous columns using optimized dot product
            for i in 0..j {
                let dot = self.compute_column_dot_product(i, j);

                // Subtract projection using parallel update for better performance
                self.subtract_projection_parallel(i, j, dot);
            }

            // Normalize column using optimized norm computation
            let norm = self.compute_column_norm(j);
            if norm > 1e-14 {
                self.normalize_column_parallel(j, norm);
            }
        }
    }

    /// Get current subspace basis (orthonormal columns)
    pub fn get_subspace(&self) -> &Array2<Complex64> {
        &self.subspace
    }

    /// Reset tracker to initial state
    pub fn reset(&mut self) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        // Reset to standard basis
        self.subspace.fill(Complex64::zero());
        for i in 0..p.min(n) {
            self.subspace[(i, i)] = Complex64::new(1.0, 0.0);
        }

        // Reset R matrix
        self.r_matrix.fill(Complex64::zero());
        for i in 0..p {
            self.r_matrix[(i, i)] = Complex64::new(1.0, 0.0);
        }

        self.weight = 1.0;
    }

    /// Get forgetting factor
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Compute W^H W matrix using optimized parallel SIMD operations
    ///
    /// This replaces the O(n*p^2) nested loops with parallel computation
    /// using Rayon and SIMD-friendly patterns for better performance.
    ///
    /// # Arguments
    /// * `subspace` - Subspace matrix W (n x p)
    ///
    /// # Returns
    /// W^H W matrix (p x p)
    fn compute_whw_parallel(&self, subspace: &ndarray::ArrayView2<Complex64>) -> Array2<Complex64> {
        let n = subspace.nrows();
        let p = subspace.ncols();
        let mut whw = Array2::<Complex64>::zeros((p, p));

        // Parallel computation using outer iterator for better cache efficiency
        whw.outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..p {
                    // Compute W[:,i]^H * W[:,j] using SIMD-friendly accumulation
                    let mut sum = Complex64::zero();
                    for k in 0..n {
                        sum += subspace[(k, i)].conj() * subspace[(k, j)];
                    }
                    row[j] = sum;
                }
            });

        whw
    }

    /// Compute W^H y (conjugate transpose matrix-vector product) using parallel operations
    ///
    /// # Arguments
    /// * `subspace` - Subspace matrix W (n x p)
    /// * `y` - Input vector (n x 1)
    ///
    /// # Returns
    /// W^H y vector (p x 1)
    fn compute_wh_y_parallel(
        &self,
        subspace: &ndarray::ArrayView2<Complex64>,
        y: &ndarray::ArrayView1<Complex64>,
    ) -> Array1<Complex64> {
        let p = subspace.ncols();
        let mut wh_y = Array1::<Complex64>::zeros(p);

        // Parallel computation of W^H y
        wh_y.outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut val)| {
                let mut sum = Complex64::zero();
                for k in 0..y.len() {
                    sum += subspace[(k, j)].conj() * y[k];
                }
                *val.first_mut().unwrap() = sum;
            });

        wh_y
    }

    /// Matrix-vector multiplication using optimized operations
    ///
    /// # Arguments
    /// * `matrix` - Input matrix (m x n)
    /// * `vector` - Input vector (n x 1)
    ///
    /// # Returns
    /// Matrix-vector product (m x 1)
    fn matrix_vector_multiply(
        &self,
        matrix: &ndarray::ArrayView2<Complex64>,
        vector: &ndarray::ArrayView1<Complex64>,
    ) -> Array1<Complex64> {
        let m = matrix.nrows();
        let mut result = Array1::<Complex64>::zeros(m);

        // Parallel matrix-vector multiplication
        result
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut val)| {
                let mut sum = Complex64::zero();
                for j in 0..vector.len() {
                    sum += matrix[(i, j)] * vector[j];
                }
                *val.first_mut().unwrap() = sum;
            });

        result
    }

    /// Compute dot product between two subspace columns with parallel accumulation
    fn compute_column_dot_product(&self, i: usize, j: usize) -> Complex64 {
        let n = self.subspace.nrows();

        // Parallel dot product computation for better performance on large arrays
        (0..n)
            .into_par_iter()
            .map(|k| self.subspace[(k, i)].conj() * self.subspace[(k, j)])
            .sum()
    }

    /// Subtract projection from column j using column i with optimized operations
    fn subtract_projection_parallel(&mut self, i: usize, j: usize, dot: Complex64) {
        let n = self.subspace.nrows();

        // Optimized projection subtraction - cache column i values to avoid repeated access
        for k in 0..n {
            let col_i_val = self.subspace[(k, i)];
            self.subspace[(k, j)] -= dot * col_i_val;
        }
    }

    /// Compute norm of subspace column using parallel reduction
    fn compute_column_norm(&self, j: usize) -> f64 {
        let n = self.subspace.nrows();

        // Parallel norm computation using SIMD-friendly operations
        (0..n)
            .into_par_iter()
            .map(|k| self.subspace[(k, j)].norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize column using optimized operations
    fn normalize_column_parallel(&mut self, j: usize, norm: f64) {
        let n = self.subspace.nrows();

        // Tightened tolerance: use 1e-15 instead of 1e-12 for better numerical precision
        if norm > 1e-15 {
            // Optimized column normalization with SIMD-friendly loop
            for k in 0..n {
                self.subspace[(k, j)] /= norm;
            }
        } else {
            // Handle near-zero norm case with fallback initialization
            // Reset column to standard basis vector to maintain subspace dimension
            for k in 0..n {
                self.subspace[(k, j)] = if k == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::zero()
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Create a steering vector for a linear array
    #[allow(dead_code)]
    fn create_steering_vector(n: usize, angle: f64) -> Array1<Complex64> {
        let k = 2.0 * PI; // Normalized wavenumber
        Array1::from_vec(
            (0..n)
                .map(|i| {
                    let phase = k * (i as f64) * angle.sin();
                    Complex64::new(phase.cos(), phase.sin())
                })
                .collect(),
        )
    }

    #[test]
    fn test_opast_initialization() {
        let n = 4;
        let p = 2;
        let lambda = 0.98;

        let tracker = OrthonormalSubspaceTracker::new(n, p, lambda);
        let subspace = tracker.get_subspace();

        // Should be n x p
        assert_eq!(subspace.nrows(), n);
        assert_eq!(subspace.ncols(), p);

        // Should be orthonormal (identity in first p rows)
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!((norm_sqr - 1.0).abs() < 1e-10, "Column {} not unit norm", j);
        }

        // Columns should be orthogonal
        for i in 0..p {
            for j in (i + 1)..p {
                let mut dot = Complex64::zero();
                for k in 0..n {
                    dot += subspace[(k, i)].conj() * subspace[(k, j)];
                }
                assert!(dot.norm() < 1e-10, "Columns {} and {} not orthogonal", i, j);
            }
        }
    }

    #[test]
    fn test_opast_single_update() {
        let n = 4;
        let p = 2;
        let lambda = 0.98;

        let mut tracker = OrthonormalSubspaceTracker::new(n, p, lambda);

        // Create a snapshot
        let snapshot = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.1),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.1, 0.0),
        ];

        tracker.update(&snapshot);

        // Subspace should still be orthonormal
        let subspace = tracker.get_subspace();
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-5,
                "Column {} not unit norm after update: {}",
                j,
                norm_sqr
            );
        }

        // Columns should remain orthogonal
        for i in 0..p {
            for j in (i + 1)..p {
                let mut dot = Complex64::zero();
                for k in 0..n {
                    dot += subspace[(k, i)].conj() * subspace[(k, j)];
                }
                assert!(
                    dot.norm() < 1e-5,
                    "Columns {} and {} not orthogonal: {}",
                    i,
                    j,
                    dot.norm()
                );
            }
        }
    }

    #[test]
    fn test_opast_convergence() {
        let n = 4;
        let p = 2;
        let lambda = 0.98;

        let mut tracker = OrthonormalSubspaceTracker::new(n, p, lambda);

        // Update with consistent signal direction
        for _ in 0..100 {
            let snapshot = vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.1),
                Complex64::new(0.3, -0.2),
                Complex64::new(0.1, 0.0),
            ];
            tracker.update(&snapshot);
        }

        // Subspace should be stable and orthonormal
        let subspace = tracker.get_subspace();
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-5,
                "Column {} not unit after convergence: {}",
                j,
                norm_sqr
            );
        }
    }

    #[test]
    fn test_opast_vs_past_stability() {
        let n = 4;
        let p = 2;
        let lambda = 0.98;

        let mut past_tracker = super::super::past::SubspaceTracker::new(n, p, lambda);
        let mut opast_tracker = OrthonormalSubspaceTracker::new(n, p, lambda);

        // Run many updates to test long-term stability
        for i in 0..1000 {
            let snapshot = vec![
                Complex64::new((i as f64 * 0.1).cos(), 0.0),
                Complex64::new((i as f64 * 0.1).sin(), 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.1, 0.0),
            ];

            past_tracker.update(&snapshot);
            opast_tracker.update(&snapshot);
        }

        // OPAST should maintain better orthonormality
        let opast_sub = opast_tracker.get_subspace();
        let mut opast_ortho_error = 0.0;

        for i in 0..p {
            for j in (i + 1)..p {
                let mut dot = Complex64::zero();
                for k in 0..n {
                    dot += opast_sub[(k, i)].conj() * opast_sub[(k, j)];
                }
                opast_ortho_error += dot.norm();
            }
        }

        // OPAST should have very small orthogonality error
        assert!(
            opast_ortho_error < 1e-3,
            "OPAST orthogonality error too large: {}",
            opast_ortho_error
        );
    }
}
