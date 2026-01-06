//! Projection Approximation Subspace Tracking (PAST) algorithm
//!
//! This module implements the PAST algorithm for efficient subspace tracking
//! of time-varying covariance matrices.
//!
//! # Strict SSOT enforcement
//! PAST depends on small-matrix inversion during its update recursion. Until the PAST path is
//! fully migrated to SSOT `crate::utils::linear_algebra` (with explicit error propagation and no
//! silent fallbacks), this module is **feature-gated**.
//!
//! Enable with `--features legacy_algorithms` if you explicitly accept legacy numerics here.
#![cfg(feature = "legacy_algorithms")]

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

/// Recursive Subspace Tracking using PAST algorithm
///
/// Projection Approximation Subspace Tracking (PAST) efficiently tracks
/// the principal subspace of a time-varying covariance matrix.
///
/// # References
/// - Yang (1995), "Projection approximation subspace tracking"
/// - Badeau et al. (2008), "Fast multilinear singular value decomposition for structured tensors"
#[derive(Debug, Clone)]
pub struct SubspaceTracker {
    /// Subspace basis matrix (n x p)
    subspace: Array2<Complex64>,
    /// Forgetting factor (0 < lambda < 1)
    /// Typical: 0.95-0.99
    lambda: f64,
    /// Accumulated weight for normalization
    weight: f64,
}

impl SubspaceTracker {
    /// Create new subspace tracker
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

        Self {
            subspace,
            lambda,
            weight: 1.0,
        }
    }

    /// Update subspace with new data snapshot
    ///
    /// Implements PAST recursion:
    /// W(t+1) = W(t) + (y(t) - W(t)α(t)) α(t)^H
    /// where α(t) = (W(t)^H W(t))^{-1} W(t)^H y(t)
    pub fn update(&mut self, snapshot: &[Complex64]) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        assert_eq!(snapshot.len(), n, "Snapshot size mismatch");

        // Convert snapshot to Array1
        let y = Array1::from(snapshot.to_vec());

        // Compute projection coefficients: α = (W^H W)^{-1} W^H y
        let mut whw = Array2::<Complex64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let mut sum = Complex64::zero();
                for k in 0..n {
                    sum += self.subspace[(k, i)].conj() * self.subspace[(k, j)];
                }
                whw[(i, j)] = sum;
            }
        }

        // Invert W^H W (small p x p matrix)
        let whw_inv = match super::matrix_utils::invert_matrix(&whw) {
            Some(inv) => inv,
            None => {
                // Fallback: use pseudo-inverse via diagonal loading
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

        // Compute W^H y
        let mut why = Array1::<Complex64>::zeros(p);
        for i in 0..p {
            let mut sum = Complex64::zero();
            for k in 0..n {
                sum += self.subspace[(k, i)].conj() * y[k];
            }
            why[i] = sum;
        }

        // Compute α = (W^H W)^{-1} W^H y
        let mut alpha = Array1::<Complex64>::zeros(p);
        for i in 0..p {
            let mut sum = Complex64::zero();
            for j in 0..p {
                sum += whw_inv[(i, j)] * why[j];
            }
            alpha[i] = sum;
        }

        // Compute error: e = y - W α
        let mut error = y.clone();
        for k in 0..n {
            let mut sum = Complex64::zero();
            for j in 0..p {
                sum += self.subspace[(k, j)] * alpha[j];
            }
            error[k] -= sum;
        }

        // PAST update: W(t+1) = λ W(t) + e α^H
        // Apply forgetting factor
        for i in 0..n {
            for j in 0..p {
                self.subspace[(i, j)] =
                    self.lambda * self.subspace[(i, j)] + error[i] * alpha[j].conj();
            }
        }

        // Gram-Schmidt orthonormalization to maintain numerical stability
        self.orthonormalize();

        // Update weight
        self.weight = self.lambda * self.weight + 1.0;
    }

    /// Get current subspace basis
    pub fn get_subspace(&self) -> &Array2<Complex64> {
        &self.subspace
    }

    /// Gram-Schmidt orthonormalization of subspace columns
    fn orthonormalize(&mut self) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        for j in 0..p {
            // Orthogonalize against previous columns
            for k in 0..j {
                let mut dot = Complex64::zero();
                for i in 0..n {
                    dot += self.subspace[(i, k)].conj() * self.subspace[(i, j)];
                }
                // Store column k values to avoid borrow checker issues
                let col_k: Vec<Complex64> = (0..n).map(|i| self.subspace[(i, k)]).collect();
                for (i, &val) in col_k.iter().enumerate() {
                    self.subspace[(i, j)] -= dot * val;
                }
            }

            // Normalize
            let mut norm_sqr = 0.0;
            for i in 0..n {
                norm_sqr += self.subspace[(i, j)].norm_sqr();
            }
            let norm = norm_sqr.sqrt();
            if norm > 1e-12 {
                for i in 0..n {
                    self.subspace[(i, j)] /= norm;
                }
            }
        }
    }

    /// Reset tracker to initial state
    pub fn reset(&mut self) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        self.subspace = Array2::<Complex64>::zeros((n, p));
        for i in 0..p.min(n) {
            self.subspace[(i, i)] = Complex64::new(1.0, 0.0);
        }
        self.weight = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Create a steering vector for a linear array
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
    fn test_subspace_tracker_initialization() {
        let n = 4;
        let p = 2; // Track 2-dimensional subspace

        let tracker = SubspaceTracker::new(n, p, 0.99);
        let subspace = tracker.get_subspace();

        // Should be identity-like initially
        assert_eq!(subspace.nrows(), n);
        assert_eq!(subspace.ncols(), p);

        // Columns should be orthonormal
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-10,
                "Column {} not unit: {}",
                j,
                norm_sqr
            );
        }
    }

    #[test]
    fn test_subspace_tracker_update() {
        let n = 4;
        let p = 2;
        let mut tracker = SubspaceTracker::new(n, p, 0.98);

        // Simulate signal from direction 30 degrees
        let angle = 30.0_f64.to_radians();
        let signal = create_steering_vector(n, angle);

        // Add some noise
        let mut snapshot = signal.clone();
        snapshot[0] += Complex64::new(0.01, 0.01);
        snapshot[1] += Complex64::new(-0.01, 0.005);

        // Update tracker - convert Array1 to slice
        let snapshot_slice: Vec<Complex64> = snapshot.iter().copied().collect();
        tracker.update(&snapshot_slice);

        // Get updated subspace
        let subspace = tracker.get_subspace();

        // Should still be orthonormal
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-6,
                "Column {} not unit norm after update: {}",
                j,
                norm_sqr
            );
        }
    }

    #[test]
    fn test_subspace_tracker_convergence() {
        let n = 6;
        let p = 3;
        let mut tracker = SubspaceTracker::new(n, p, 0.95);

        // Create a consistent signal
        let signal = create_steering_vector(n, 45.0_f64.to_radians());

        // Apply many updates
        for i in 0..50 {
            let mut snapshot = signal.clone();
            // Small noise
            for j in 0..n {
                snapshot[j] += Complex64::new(
                    ((i * j) as f64 * 0.001).sin() * 0.01,
                    ((i * j) as f64 * 0.001).cos() * 0.01,
                );
            }
            let snapshot_slice: Vec<Complex64> = snapshot.iter().copied().collect();
            tracker.update(&snapshot_slice);
        }

        // Subspace should still be valid
        let subspace = tracker.get_subspace();
        assert!(subspace.iter().all(|x| x.is_finite()));

        // Should still be orthonormal
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
}
