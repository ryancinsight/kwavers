//! Advanced Beamforming Algorithms
//!
//! This module implements state-of-the-art beamforming algorithms including:
//! - Delay-and-Sum (conventional beamforming)
//! - MVDR (Capon beamformer - Minimum Variance Distortionless Response)
//! - MUSIC (Multiple Signal Classification)
//! - Eigenspace-based Minimum Variance (ESPMV)
//!
//! # References
//! - Capon (1969), "High-resolution frequency-wavenumber spectrum analysis"
//! - Schmidt (1986), "Multiple emitter location and signal parameter estimation"
//! - Van Trees (2002), "Optimum Array Processing"
//! - Stoica & Nehorai (1990), "MUSIC, maximum likelihood, and Cramer-Rao bound"

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

/// Simple matrix inversion using Gauss-Jordan elimination
/// Returns None if matrix is singular
fn invert_matrix(mat: &Array2<Complex64>) -> Option<Array2<Complex64>> {
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
fn eigen_hermitian(mat: &Array2<Complex64>, num_eigs: usize) -> Option<(Vec<f64>, Array2<Complex64>)> {
    let n = mat.nrows();
    if n != mat.ncols() || num_eigs == 0 || num_eigs > n {
        return None;
    }

    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Array2::<Complex64>::zeros((n, num_eigs));
    let mut a = mat.clone();

    for col in 0..num_eigs {
        // Power iteration for current eigenvalue
        let mut v = Array1::<Complex64>::from_vec((0..n).map(|i| Complex64::new((i + 1) as f64, 0.0)).collect());
        
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
            let diff: f64 = v.iter().zip(v_new.iter()).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
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

/// Beamforming algorithm trait
pub trait BeamformingAlgorithm {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64>;
}

/// Delay and sum beamforming (conventional beamforming)
///
/// The simplest beamforming algorithm that applies uniform weighting
/// to all array elements after steering delays.
///
/// # References
/// - Van Veen & Buckley (1988), "Beamforming: A versatile approach to spatial filtering"
#[derive(Debug)]
pub struct DelayAndSum;

impl BeamformingAlgorithm for DelayAndSum {
    fn compute_weights(
        &self,
        _covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        // Conventional beamforming: w = a (steering vector)
        steering.clone()
    }
}

/// Minimum Variance Distortionless Response (MVDR / Capon) beamformer
///
/// The MVDR beamformer minimizes output power while maintaining unit gain
/// in the look direction. Also known as the Capon beamformer.
///
/// The weight vector is: w = R^{-1} a / (a^H R^{-1} a)
///
/// where:
/// - R is the sample covariance matrix
/// - a is the steering vector
///
/// # References
/// - Capon (1969), "High-resolution frequency-wavenumber spectrum analysis",
///   Proceedings of the IEEE, 57(8), 1408-1418
/// - Van Trees (2002), "Optimum Array Processing", Ch. 6
#[derive(Debug)]
pub struct MinimumVariance {
    /// Diagonal loading factor for numerical stability
    pub diagonal_loading: f64,
}

impl Default for MinimumVariance {
    fn default() -> Self {
        Self {
            diagonal_loading: 1e-6, // Small regularization
        }
    }
}

impl MinimumVariance {
    /// Create MVDR beamformer with custom diagonal loading
    #[must_use]
    pub fn with_diagonal_loading(diagonal_loading: f64) -> Self {
        Self { diagonal_loading }
    }
}

impl BeamformingAlgorithm for MinimumVariance {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        // Add diagonal loading for numerical stability: R_loaded = R + δI
        let n = covariance.nrows();
        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(self.diagonal_loading, 0.0);
        }

        // Compute R^{-1}
        let r_inv = match invert_matrix(&r_loaded) {
            Some(inv) => inv,
            None => {
                // Fallback to delay-and-sum if inversion fails
                return steering.clone();
            }
        };

        // Compute R^{-1} a
        let r_inv_a = r_inv.dot(steering);

        // Compute a^H R^{-1} a
        let a_h_r_inv_a: Complex64 = steering
            .iter()
            .zip(r_inv_a.iter())
            .map(|(a, r)| a.conj() * r)
            .sum();

        // Avoid division by zero
        if a_h_r_inv_a.norm() < 1e-12 {
            return steering.clone();
        }

        // w = R^{-1} a / (a^H R^{-1} a)
        r_inv_a.mapv(|x| x / a_h_r_inv_a)
    }
}

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
    use approx::assert_relative_eq;
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
    fn test_delay_and_sum() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = DelayAndSum;
        let weights = beamformer.compute_weights(&cov, &steering);

        // Weights should equal steering vector
        for i in 0..n {
            assert_relative_eq!(weights[i].re, steering[i].re, epsilon = 1e-10);
            assert_relative_eq!(weights[i].im, steering[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mvdr_weights_exist() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = MinimumVariance::default();
        let weights = beamformer.compute_weights(&cov, &steering);

        // Weights should be finite
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_mvdr_unit_gain_constraint() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = MinimumVariance::default();
        let weights = beamformer.compute_weights(&cov, &steering);

        // Check unit gain constraint: w^H a = 1
        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(w, a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.norm(), 1.0, epsilon = 1e-6);
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

        let mvdr = MinimumVariance::default();
        let espmv = EigenspaceMV::new(n); // Full rank = equivalent to MVDR

        let weights_mvdr = mvdr.compute_weights(&cov, &steering);
        let weights_espmv = espmv.compute_weights(&cov, &steering);

        // Should be similar for full-rank signal subspace
        let diff_norm: f64 = weights_mvdr
            .iter()
            .zip(weights_espmv.iter())
            .map(|(w1, w2)| (w1 - w2).norm())
            .sum();

        // Allow some difference due to numerical precision
        assert!(diff_norm < 1.0, "Difference: {}", diff_norm);
    }

    #[test]
    fn test_mvdr_diagonal_loading() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = MinimumVariance::with_diagonal_loading(1e-3);
        let weights = beamformer.compute_weights(&cov, &steering);

        // Should still produce valid weights
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }
}
