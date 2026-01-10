//! Robust Capon Beamformer (RCB)
//!
//! # SSOT / correctness stance
//! This implementation must not perform local matrix inversion and must not silently mask numerical
//! failures. It uses the SSOT complex linear solver:
//! `crate::utils::linear_algebra::LinearAlgebra::solve_linear_system_complex`.
//!
//! # Mathematical definition
//! RCB here is implemented as MVDR with a robustness-informed diagonal loading factor:
//!
//! `w = R_loaded^{-1} a / (aᴴ R_loaded^{-1} a)`
//!
//! where `R_loaded = R + δ I`, and `δ` is selected using a mismatch/uncertainty heuristic.
//!
//! # Error policy (no masking)
//! - Dimension mismatches, singular systems, or invalid denominators are explicit errors.
//! - No fallback to `steering.clone()` is permitted.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

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
    ) -> KwaversResult<Array1<Complex64>> {
        let n = covariance.nrows();
        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(format!(
                "RobustCapon: covariance must be a non-empty square matrix; got {}x{}",
                covariance.nrows(),
                covariance.ncols()
            )));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(format!(
                "RobustCapon: steering length ({}) must match covariance dimension ({n})",
                steering.len()
            )));
        }

        // Compute adaptive loading factor
        let loading = self.compute_loading_factor(covariance, steering);
        if !loading.is_finite() || loading < 0.0 {
            return Err(KwaversError::InvalidInput(
                "RobustCapon: computed loading must be finite and >= 0".to_string(),
            ));
        }

        // Apply diagonal loading: R_loaded = R + δI
        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(loading, 0.0);
        }

        // Solve R_loaded y = a (SSOT complex solve). Avoid explicit inversion.
        let y = LinearAlgebra::solve_linear_system_complex(&r_loaded, steering)?;

        // denom = aᴴ y (should be real-positive for Hermitian PD covariance with δ > 0)
        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, yi): (_, _)| a.conj() * yi)
            .sum();

        let denom_re = denom.re;
        if !denom_re.is_finite() || denom_re <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::domain::core::error::NumericalError::InvalidOperation(
                    "RobustCapon: non-positive or non-finite denominator a^H R^{-1} a".to_string(),
                ),
            ));
        }

        // w = y / (aᴴ y)
        Ok(y.mapv(|x| x / denom_re))
    }
}
