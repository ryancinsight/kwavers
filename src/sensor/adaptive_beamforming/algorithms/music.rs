//! MUSIC (Multiple Signal Classification) algorithm

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

use super::utils::eigen_hermitian;
use super::BeamformingAlgorithm;

/// MUSIC (Multiple Signal Classification) algorithm
///
/// MUSIC is a subspace-based method that exploits the eigenstructure of
/// the covariance matrix to estimate directions of arrival (DOA).
///
/// The MUSIC pseudospectrum is: P_MUSIC(θ) = 1 / (a^H P_N P_N^H a)
///
/// where:
/// - P_N is the projection onto the noise subspace
/// - a is the steering vector
///
/// # References
/// - Schmidt (1986), "Multiple emitter location and signal parameter estimation",
///   IEEE Transactions on Antennas and Propagation, 34(3), 276-280
/// - Stoica & Nehorai (1990), "MUSIC, maximum likelihood, and Cramer-Rao bound",
///   IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(5), 720-741
#[derive(Debug)]
pub struct MUSIC {
    /// Number of sources (signals)
    pub num_sources: usize,
}

impl MUSIC {
    /// Create MUSIC algorithm with specified number of sources
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        Self { num_sources }
    }

    /// Compute MUSIC pseudospectrum value for given steering vector
    ///
    /// # Arguments
    /// * `covariance` - Sample covariance matrix
    /// * `steering` - Steering vector for direction of interest
    ///
    /// # Returns
    /// MUSIC pseudospectrum value (higher = more likely source direction)
    pub fn pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> f64 {
        // Eigendecomposition of covariance matrix
        let n = covariance.nrows();
        let (eigenvalues, eigenvectors) = match eigen_hermitian(covariance, n) {
            Some((vals, vecs)) => (vals, vecs),
            None => return 0.0, // Fallback
        };

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Noise subspace: eigenvectors corresponding to smallest eigenvalues
        let n = covariance.nrows();
        let noise_start = self.num_sources.min(n);

        // Build noise subspace projection: P_N = Σ e_i e_i^H
        let mut p_n = Array2::<Complex64>::zeros((n, n));
        for &idx in indices.iter().skip(noise_start) {
            let e_i = eigenvectors.column(idx);
            for i in 0..n {
                for j in 0..n {
                    p_n[(i, j)] += e_i[i] * e_i[j].conj();
                }
            }
        }

        // Compute a^H P_N a
        let mut a_h_pn_a = Complex64::zero();
        for i in 0..n {
            for j in 0..n {
                a_h_pn_a += steering[i].conj() * p_n[(i, j)] * steering[j];
            }
        }

        // MUSIC pseudospectrum: 1 / |a^H P_N a|
        let denominator = a_h_pn_a.norm();
        if denominator < 1e-12 {
            0.0
        } else {
            1.0 / denominator
        }
    }
}

impl BeamformingAlgorithm for MUSIC {
    fn compute_weights(
        &self,
        _covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        // MUSIC doesn't directly provide weights; return steering vector
        // In practice, MUSIC is used for DOA estimation, not beamforming
        steering.clone()
    }
}
