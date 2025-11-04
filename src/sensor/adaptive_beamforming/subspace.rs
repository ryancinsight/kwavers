//! Subspace-based beamforming algorithms
//!
//! This module implements high-resolution beamforming techniques that exploit
//! the eigenstructure of the covariance matrix for improved performance.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

use super::conventional::BeamformingAlgorithm;
use super::matrix_utils::{eigen_hermitian, invert_matrix};
use super::source_estimation::{estimate_num_sources, SourceEstimationCriterion};

/// MUSIC (Multiple Signal Classification) algorithm
///
/// MUSIC is a subspace-based method that exploits the eigenstructure of
/// the covariance matrix to estimate directions of arrival (DOA) with
/// super-resolution capabilities.
///
/// The MUSIC pseudospectrum is: P_MUSIC(θ) = 1 / (a^H P_N P_N^H a)
///
/// where:
/// - P_N is the projection onto the noise subspace
/// - a is the steering vector for direction θ
///
/// ## Usage
///
/// MUSIC automatically estimates the number of sources using AIC/MDL criteria,
/// then performs eigendecomposition to separate signal and noise subspaces.
///
/// ```rust
/// use kwavers::sensor::adaptive_beamforming::{MUSIC, SourceEstimationCriterion};
/// use ndarray::Array2;
/// use num_complex::Complex64;
///
/// // Covariance matrix from signal processing
/// let cov = Array2::<Complex64>::eye(8);
/// let num_snapshots = 100; // Number of snapshots used to compute covariance
///
/// // Create MUSIC beamformer with automatic source estimation
/// let music = MUSIC::new_with_source_estimation(&cov, num_snapshots, SourceEstimationCriterion::MDL);
///
/// // Or create with manual source count
/// let music = MUSIC::new(2);
///
/// // Compute pseudospectra for many steering directions:
/// // for angle in steering_angles {
/// //     let steering = compute_steering_vector(angle);
/// //     let spectrum = music.pseudospectrum(&cov, &steering);
/// //     // Peak detection gives DOA estimates
/// // }
/// ```
///
/// ## Algorithm Details
///
/// 1. **Source Estimation**: Automatic M detection using AIC/MDL criteria
/// 2. **Eigendecomposition**: R = U Σ U^H
/// 3. **Subspace Separation**: Signal subspace U_s (first M eigenvectors)
/// 4. **Noise Subspace**: U_n (remaining eigenvectors)
/// 5. **Pseudospectrum**: P(θ) = 1 / ||U_n^H a(θ)||²
///
/// ## Performance
///
/// - **Time Complexity**: O(N³) for eigendecomposition (dominant)
/// - **Space Complexity**: O(N²) for eigenvector storage
/// - **Resolution**: Can resolve sources closer than λ/(2D) radians
/// - **Robustness**: Automatic source estimation improves reliability
///
/// ## When to Use
///
/// - High-resolution DOA estimation required
/// - Multiple sources with small angular separation
/// - Off-line processing (not real-time due to complexity)
/// - Well-conditioned covariance matrices
///
/// # References
/// - Schmidt (1986), "Multiple emitter location and signal parameter estimation",
///   IEEE Transactions on Antennas and Propagation, 34(3), 276-280
/// - Stoica & Nehorai (1990), "MUSIC, maximum likelihood, and Cramer-Rao bound",
///   IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(5), 720-741
/// - Wax & Kailath (1985), "Detection of signals by information theoretic criteria",
///   IEEE Transactions on Acoustics, Speech, and Signal Processing, 33(2), 387-392
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

    /// Create MUSIC algorithm with automatic source number estimation
    ///
    /// Uses AIC or MDL criteria to automatically determine the number of sources
    /// from the covariance matrix eigenvalues.
    ///
    /// # Arguments
    /// * `covariance` - Sample covariance matrix
    /// * `num_snapshots` - Number of temporal snapshots used to compute covariance
    /// * `criterion` - Information criterion (AIC or MDL)
    ///
    /// # Returns
    /// MUSIC algorithm with automatically estimated source count
    ///
    /// # References
    /// - Wax & Kailath (1985), "Detection of signals by information theoretic criteria"
    #[must_use]
    pub fn new_with_source_estimation(
        covariance: &Array2<Complex64>,
        num_snapshots: usize,
        criterion: SourceEstimationCriterion,
    ) -> Self {
        let num_sources = estimate_num_sources(covariance, num_snapshots, criterion);
        Self::new(num_sources)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::f64::consts::PI;

    /// Create a simple test covariance matrix
    fn create_test_covariance(n: usize) -> Array2<Complex64> {
        let mut r = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let val = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.1 / (1.0 + (i as f64 - j as f64).abs()), 0.0)
                };
                r[(i, j)] = val;
            }
        }
        r
    }

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
    fn test_music_pseudospectrum() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let music = MUSIC::new(1); // 1 source
        let spectrum = music.pseudospectrum(&cov, &steering);

        // Pseudospectrum should be positive
        assert!(spectrum >= 0.0);
        assert!(spectrum.is_finite());
    }

    #[test]
    fn test_music_peak_detection() {
        let n = 8;
        let cov = create_test_covariance(n);
        let music = MUSIC::new(1);

        // Scan angles and find peak
        let angles: Vec<f64> = (0..180).map(|i| (i as f64 - 90.0) * PI / 180.0).collect();
        let spectrum: Vec<f64> = angles
            .iter()
            .map(|&angle| {
                let steering = create_steering_vector(n, angle);
                music.pseudospectrum(&cov, &steering)
            })
            .collect();

        // Should have a peak somewhere
        let max_val = spectrum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.0);
        assert!(max_val.is_finite());
    }

    #[test]
    fn test_eigenspace_mv() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = EigenspaceMV::new(2); // 2 sources
        let weights = beamformer.compute_weights(&cov, &steering);

        // Weights should be finite
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_eigenspace_mv_vs_mvdr() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let mvdr = crate::sensor::adaptive_beamforming::MinimumVariance::default();
        let espmv = EigenspaceMV::new(n); // Full rank = equivalent to MVDR

        let weights_mvdr = mvdr.compute_weights(&cov, &steering);
        let weights_espmv = espmv.compute_weights(&cov, &steering);

        // Should be similar for full-rank signal subspace
        let diff_norm: f64 = weights_mvdr
            .iter()
            .zip(weights_espmv.iter())
            .map(|(w1, w2)| (*w1 - *w2).norm())
            .sum();

        // Allow some difference due to numerical precision
        assert!(diff_norm < 1.0, "Difference: {}", diff_norm);
    }

    #[test]
    fn test_music_with_source_estimation() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        // Test with AIC criterion
        let music_aic = MUSIC::new_with_source_estimation(
            &cov,
            num_snapshots,
            SourceEstimationCriterion::AIC,
        );

        // Test with MDL criterion
        let music_mdl = MUSIC::new_with_source_estimation(
            &cov,
            num_snapshots,
            SourceEstimationCriterion::MDL,
        );

        // Both should have valid source counts
        assert!(music_aic.num_sources < n);
        assert!(music_mdl.num_sources < n);

        // MDL should be more conservative (fewer sources) than AIC
        assert!(music_mdl.num_sources <= music_aic.num_sources);

        // Test that pseudospectrum computation works
        let steering = create_steering_vector(n, 0.0);
        let spectrum_aic = music_aic.pseudospectrum(&cov, &steering);
        let spectrum_mdl = music_mdl.pseudospectrum(&cov, &steering);

        assert!(spectrum_aic >= 0.0);
        assert!(spectrum_mdl >= 0.0);
        assert!(spectrum_aic.is_finite());
        assert!(spectrum_mdl.is_finite());
    }
}
