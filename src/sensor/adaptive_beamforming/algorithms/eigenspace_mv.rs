//! Eigenspace-based Minimum Variance (ESPMV) beamformer

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

use super::utils::{eigen_hermitian, invert_matrix};
use super::BeamformingAlgorithm;

/// Eigenspace-based Minimum Variance (ESPMV) beamformer
///
/// ESPMV is a robust beamformer that operates in the signal subspace,
/// reducing sensitivity to noise and model errors.
///
/// The weight vector is: w = P_S R^{-1} a / (a^H R^{-1} P_S a)
///
/// where P_S is the projection onto the signal subspace.
///
/// # References
/// - Gershman et al. (1999), "Adaptive beamforming algorithms with robustness
///   against jammer motion", IEEE Transactions on Signal Processing
/// - Shahbazpanahi et al. (2003), "A generalized Capon estimator for localization
///   of multiple spread sources", IEEE Transactions on Signal Processing
#[derive(Debug)]
pub struct EigenspaceMV {
    /// Number of sources (signal subspace dimension)
    pub num_sources: usize,
    /// Diagonal loading factor
    pub diagonal_loading: f64,
}

impl EigenspaceMV {
    /// Create Eigenspace MV beamformer
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        Self {
            num_sources,
            diagonal_loading: 1e-6,
        }
    }

    /// Create with custom diagonal loading
    #[must_use]
    pub fn with_diagonal_loading(num_sources: usize, diagonal_loading: f64) -> Self {
        Self {
            num_sources,
            diagonal_loading,
        }
    }
}

impl BeamformingAlgorithm for EigenspaceMV {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        let n = covariance.nrows();

        // Add diagonal loading
        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(self.diagonal_loading, 0.0);
        }

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = match eigen_hermitian(&r_loaded, n) {
            Some((vals, vecs)) => (vals, vecs),
            None => return steering.clone(), // Fallback
        };

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Signal subspace: eigenvectors corresponding to largest eigenvalues
        let num_signal = self.num_sources.min(n);

        // Build signal subspace projection: P_S = Σ e_i e_i^H
        let mut p_s = Array2::<Complex64>::zeros((n, n));
        for &idx in indices.iter().take(num_signal) {
            let e_i = eigenvectors.column(idx);
            for i in 0..n {
                for j in 0..n {
                    p_s[(i, j)] += e_i[i] * e_i[j].conj();
                }
            }
        }

        // Compute R^{-1}
        let r_inv = match invert_matrix(&r_loaded) {
            Some(inv) => inv,
            None => return steering.clone(),
        };

        // Compute P_S R^{-1} a
        let r_inv_a = r_inv.dot(steering);
        let mut ps_r_inv_a = Array1::<Complex64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                ps_r_inv_a[i] += p_s[(i, j)] * r_inv_a[j];
            }
        }

        // Compute a^H R^{-1} P_S a
        let mut a_h_r_inv_ps_a = Complex64::zero();
        for i in 0..n {
            a_h_r_inv_ps_a += steering[i].conj() * ps_r_inv_a[i];
        }

        // Avoid division by zero
        if a_h_r_inv_ps_a.norm() < 1e-12 {
            return steering.clone();
        }

        // w = P_S R^{-1} a / (a^H R^{-1} P_S a)
        ps_r_inv_a.mapv(|x| x / a_h_r_inv_ps_a)
    }
}
