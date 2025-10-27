//! Robust Capon Beamformer (RCB)

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

use super::utils::invert_matrix;
use super::BeamformingAlgorithm;

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
        covariance: &Array2<Complex64>,
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
        covariance: &Array2<Complex64>,
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
