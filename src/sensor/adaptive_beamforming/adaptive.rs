//! Adaptive beamforming algorithms
//!
//! This module implements adaptive beamforming techniques that optimize
//! performance based on the received data statistics.

use ndarray::Array1;
use num_complex::Complex64;
use num_traits::Zero;

use super::conventional::BeamformingAlgorithm;
use super::matrix_utils::invert_matrix;

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
/// ## Usage
///
/// ```rust
/// use kwavers::sensor::adaptive_beamforming::MinimumVariance;
///
/// // Create MVDR beamformer with default diagonal loading (1e-6)
/// let beamformer = MinimumVariance::default();
///
/// // Or specify custom diagonal loading for numerical stability
/// let custom_beamformer = MinimumVariance::with_diagonal_loading(1e-4);
///
/// // In practice, you would then use this with a covariance matrix and steering vector:
/// // let weights = beamformer.compute_weights(&covariance, &steering);
/// ```
///
/// ## Performance
///
/// - **Time Complexity**: O(N³) due to matrix inversion
/// - **Space Complexity**: O(N²) for covariance matrix storage
/// - **Suitable for**: Arrays with N ≤ 32 elements in real-time applications
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
        covariance: &ndarray::Array2<Complex64>,
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

/// Robust Capon Beamformer (RCB)
///
/// The Robust Capon Beamformer addresses the sensitivity of MVDR to steering vector
/// errors and array calibration uncertainties. It optimizes for worst-case performance
/// over an uncertainty set.
///
/// Uses diagonal loading with automatic loading factor selection based on:
/// - Array geometry uncertainty
/// - Desired robustness level
///
/// # References
/// - Vorobyov et al. (2003), "Robust adaptive beamforming using worst-case performance
///   optimization: A solution to the signal mismatch problem", IEEE Trans. SP, 51(2), 313-324
/// - Li et al. (2003), "On robust Capon beamforming and diagonal loading",
///   IEEE Transactions on Signal Processing, 51(7), 1702-1715
/// - Lorenz & Boyd (2005), "Robust minimum variance beamforming",
///   IEEE Transactions on Signal Processing, 53(5), 1684-1696
#[derive(Debug)]
pub struct RobustCapon {
    /// Uncertainty bound (steering vector mismatch tolerance)
    /// Typical values: 0.01 to 0.2 (1% to 20% uncertainty)
    pub uncertainty_bound: f64,
    /// Base diagonal loading factor
    pub base_loading: f64,
    /// Enable adaptive loading factor computation
    pub adaptive_loading: bool,
}

impl Default for RobustCapon {
    fn default() -> Self {
        Self {
            uncertainty_bound: 0.05, // 5% uncertainty
            base_loading: 1e-6,
            adaptive_loading: true,
        }
    }
}

impl RobustCapon {
    /// Create Robust Capon beamformer with specified uncertainty bound
    ///
    /// # Arguments
    /// * `uncertainty_bound` - Steering vector mismatch tolerance (0.0 to 1.0)
    ///   - 0.01: 1% uncertainty (precise calibration)
    ///   - 0.05: 5% uncertainty (typical)
    ///   - 0.20: 20% uncertainty (large errors)
    #[must_use]
    pub fn new(uncertainty_bound: f64) -> Self {
        Self {
            uncertainty_bound: uncertainty_bound.clamp(0.0, 1.0),
            base_loading: 1e-6,
            adaptive_loading: true,
        }
    }

    /// Create with custom base diagonal loading
    #[must_use]
    pub fn with_loading(uncertainty_bound: f64, base_loading: f64) -> Self {
        Self {
            uncertainty_bound: uncertainty_bound.clamp(0.0, 1.0),
            base_loading,
            adaptive_loading: true,
        }
    }

    /// Disable adaptive loading (use only base loading)
    #[must_use]
    pub fn without_adaptive_loading(mut self) -> Self {
        self.adaptive_loading = false;
        self
    }

    /// Compute adaptive loading factor based on uncertainty bound and covariance
    ///
    /// Uses the method from Vorobyov et al. (2003) / Li et al. (2003)
    fn compute_loading_factor(
        &self,
        covariance: &ndarray::Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> f64 {
        if !self.adaptive_loading {
            return self.base_loading;
        }

        let n = covariance.nrows();

        // Compute steering vector norm
        let a_norm_sq: f64 = steering.iter().map(|x| x.norm_sqr()).sum();

        // Estimate noise power from smallest eigenvalues
        // Quick estimation: use trace / n as approximation
        let mut trace = Complex64::zero();
        for i in 0..n {
            trace += covariance[(i, i)];
        }
        let noise_power = (trace.re / (n as f64)).max(1e-12);

        // Adaptive loading factor based on uncertainty bound
        // δ = ε * sqrt(noise_power * ||a||²)
        // where ε is the uncertainty bound
        let epsilon = self.uncertainty_bound;
        let loading = epsilon * (noise_power * a_norm_sq).sqrt();

        // Combine with base loading
        loading.max(self.base_loading)
    }
}

impl BeamformingAlgorithm for RobustCapon {
    fn compute_weights(
        &self,
        covariance: &ndarray::Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        let n = covariance.nrows();

        // Compute adaptive loading factor
        let loading = self.compute_loading_factor(covariance, steering);

        // Apply diagonal loading: R_loaded = R + δI
        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(loading, 0.0);
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
        // This is the MVDR solution with robust diagonal loading
        r_inv_a.mapv(|x| x / a_h_r_inv_a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
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

    #[test]
    fn test_robust_capon_default() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = RobustCapon::default();
        let weights = beamformer.compute_weights(&cov, &steering);

        // Should produce valid weights
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_robust_capon_unit_gain() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = RobustCapon::new(0.1); // 10% uncertainty
        let weights = beamformer.compute_weights(&cov, &steering);

        // Check unit gain constraint: w^H a ≈ 1
        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(w, a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.norm(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_robust_capon_uncertainty_bounds() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        // Test different uncertainty bounds
        for uncertainty in &[0.01, 0.05, 0.1, 0.2] {
            let beamformer = RobustCapon::new(*uncertainty);
            let weights = beamformer.compute_weights(&cov, &steering);

            assert_eq!(weights.len(), n);
            for &w in &weights {
                assert!(w.is_finite());
            }
        }
    }

    #[test]
    fn test_robust_capon_vs_mvdr() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::default();
        let rcb = RobustCapon::new(0.01); // Very small uncertainty → similar to MVDR

        let weights_mvdr = mvdr.compute_weights(&cov, &steering);
        let weights_rcb = rcb.compute_weights(&cov, &steering);

        // With small uncertainty, RCB should be similar to MVDR
        let diff_norm: f64 = weights_mvdr
            .iter()
            .zip(weights_rcb.iter())
            .map(|(w1, w2)| (w1 - w2).norm())
            .sum();

        // Should be relatively close
        assert!(diff_norm < 2.0, "Difference: {}", diff_norm);
    }
}
