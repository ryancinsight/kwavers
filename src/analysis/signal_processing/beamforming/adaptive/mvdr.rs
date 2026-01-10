//! Minimum Variance Distortionless Response (MVDR/Capon) Beamformer
//!
//! # Mathematical Foundation
//!
//! The MVDR beamformer, also known as the **Capon beamformer**, is an adaptive beamforming
//! algorithm that minimizes the output power while maintaining unit gain in the look direction.
//!
//! ## Problem Formulation
//!
//! Given:
//! - Covariance matrix **R** (N×N Hermitian positive semi-definite)
//! - Steering vector **a** (N×1 complex)
//!
//! Find weight vector **w** that solves:
//!
//! ```text
//! minimize   w^H R w
//! subject to w^H a = 1
//! ```
//!
//! ## Closed-Form Solution
//!
//! The optimal weight vector is:
//!
//! ```text
//! w = R^{-1} a / (a^H R^{-1} a)
//! ```
//!
//! where:
//! - `^H` denotes conjugate transpose (Hermitian)
//! - `R^{-1}` is the inverse of the covariance matrix
//!
//! ## Diagonal Loading
//!
//! For numerical stability and robustness, we use **diagonal loading**:
//!
//! ```text
//! R_loaded = R + δI
//! w = R_loaded^{-1} a / (a^H R_loaded^{-1} a)
//! ```
//!
//! where δ ≥ 0 is the diagonal loading factor (typically 1e-6 to 1e-3).
//!
//! ## Properties
//!
//! 1. **Unit Gain Constraint**: w^H a = 1 (distortionless response in look direction)
//! 2. **Minimum Output Power**: Minimizes interference and noise power
//! 3. **Adaptive Nulling**: Automatically forms nulls in interference directions
//! 4. **Optimal SINR**: Maximizes signal-to-interference-plus-noise ratio
//!
//! # Architectural Intent (SSOT)
//!
//! This implementation adheres to the **Single Source of Truth** (SSOT) principle:
//!
//! - **NO local matrix inversion** - uses `math::linear_algebra::LinearAlgebra::solve_linear_system_complex`
//! - **NO silent fallbacks** - returns `Err(...)` on numerical failure
//! - **NO error masking** - all failures are explicit and surfaced to caller
//! - **NO dummy weights** - never returns steering vector as disguised fallback
//!
//! This module is in the **analysis layer** (Layer 7) because:
//! 1. Beamforming is a signal processing algorithm, not a domain primitive
//! 2. Analysis layer can import from domain (sensor geometry) and math (linear algebra)
//! 3. Enables reusability across simulations, sensors, and clinical workflows
//!
//! # Literature References
//!
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//!   DOI: 10.1109/PROC.1969.7278
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection,
//!   Estimation, and Modulation Theory*. Wiley-Interscience. (Chapter 6: Optimum Beamforming)
//!
//! - Li, J., Stoica, P., & Wang, Z. (2003). "On robust Capon beamforming and diagonal loading."
//!   *IEEE Transactions on Signal Processing*, 51(7), 1702-1715.
//!   DOI: 10.1109/TSP.2003.812831
//!
//! # Migration Note
//!
//! This module was migrated from `domain::sensor::beamforming::adaptive::algorithms::mvdr`
//! to `analysis::signal_processing::beamforming::adaptive::mvdr` as part of the
//! architectural purification effort (ADR 003). The API remains unchanged.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Minimum Variance Distortionless Response (MVDR / Capon) beamformer.
///
/// The MVDR beamformer is an **adaptive** beamforming algorithm that minimizes the
/// output power while maintaining unit gain in the look direction. It is optimal in
/// the sense of maximizing the signal-to-interference-plus-noise ratio (SINR).
///
/// # Algorithm
///
/// Given covariance matrix **R** and steering vector **a**, the MVDR weights are:
///
/// ```text
/// w = R^{-1} a / (a^H R^{-1} a)
/// ```
///
/// With diagonal loading for stability:
/// ```text
/// w = (R + δI)^{-1} a / (a^H (R + δI)^{-1} a)
/// ```
///
/// # Usage Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::adaptive::MinimumVariance;
/// use ndarray::{Array1, Array2};
/// use num_complex::Complex64;
///
/// // Create covariance matrix (4x4 array)
/// let covariance = Array2::eye(4) * Complex64::new(1.0, 0.0);
///
/// // Create steering vector (4 elements)
/// let steering = Array1::from_vec(vec![
///     Complex64::new(1.0, 0.0),
///     Complex64::new(0.9, 0.1),
///     Complex64::new(0.8, 0.2),
///     Complex64::new(0.7, 0.3),
/// ]);
///
/// // Create MVDR beamformer with default diagonal loading
/// let mvdr = MinimumVariance::default();
///
/// // Compute optimal weights
/// let weights = mvdr.compute_weights(&covariance, &steering)?;
///
/// // Verify unit gain constraint: w^H a ≈ 1
/// let gain: Complex64 = weights.iter()
///     .zip(steering.iter())
///     .map(|(w, a)| w.conj() * a)
///     .sum();
/// assert!((gain.norm() - 1.0).abs() < 1e-6);
/// ```
///
/// # Properties
///
/// - **Adaptive Nulling**: Automatically forms nulls in interference directions
/// - **Optimal SINR**: Maximizes signal-to-interference-plus-noise ratio
/// - **Distortionless**: Unit gain in look direction (w^H a = 1)
/// - **Robust**: Diagonal loading prevents numerical instability
///
/// # Comparison to Delay-and-Sum
///
/// | Property | DAS | MVDR |
/// |----------|-----|------|
/// | Adaptation | None | Data-adaptive |
/// | Nulling | No | Automatic |
/// | SINR | Lower | Optimal |
/// | Computation | O(N) | O(N²) or O(N³) |
/// | Robustness | High | Requires loading |
///
/// # When to Use MVDR
///
/// - **Strong interference** present in the environment
/// - **Known array geometry** and steering vectors
/// - **Sufficient snapshots** to estimate covariance (typically N_snap > 2N)
/// - **SNR moderate to high** (low SNR may require robust variants)
///
/// # References
///
/// - Capon (1969): Original MVDR derivation
/// - Van Trees (2002): Comprehensive treatment in Chapter 6
/// - Li et al. (2003): Robust Capon beamforming with diagonal loading
#[derive(Debug, Clone)]
pub struct MinimumVariance {
    /// Diagonal loading factor for numerical stability.
    ///
    /// Typical values:
    /// - `1e-6` to `1e-8`: Light regularization (high SNR)
    /// - `1e-4` to `1e-2`: Moderate regularization (moderate SNR)
    /// - `0.01` to `0.1`: Heavy regularization (low SNR or model mismatch)
    pub diagonal_loading: f64,
}

impl Default for MinimumVariance {
    /// Create MVDR beamformer with default diagonal loading.
    ///
    /// Default: `diagonal_loading = 1e-6` (light regularization)
    fn default() -> Self {
        Self {
            diagonal_loading: 1e-6,
        }
    }
}

impl MinimumVariance {
    /// Create MVDR beamformer with no diagonal loading.
    ///
    /// **Warning**: This may fail for ill-conditioned covariance matrices.
    /// Prefer [`Self::with_diagonal_loading`] for robust operation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mvdr = MinimumVariance::new();
    /// assert_eq!(mvdr.diagonal_loading, 0.0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            diagonal_loading: 0.0,
        }
    }

    /// Create MVDR beamformer with custom diagonal loading.
    ///
    /// # Parameters
    ///
    /// - `diagonal_loading`: Regularization parameter δ ≥ 0
    ///
    /// # Guidelines
    ///
    /// - **High SNR, well-conditioned**: `1e-8` to `1e-6`
    /// - **Moderate SNR**: `1e-4` to `1e-2`
    /// - **Low SNR or model mismatch**: `0.01` to `0.1`
    /// - **Severe interference**: May need adaptive loading
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Moderate regularization for typical ultrasound imaging
    /// let mvdr = MinimumVariance::with_diagonal_loading(1e-4);
    /// ```
    #[must_use]
    pub fn with_diagonal_loading(diagonal_loading: f64) -> Self {
        Self { diagonal_loading }
    }

    /// Compute MVDR beamforming weights.
    ///
    /// # Algorithm
    ///
    /// 1. Apply diagonal loading: `R_loaded = R + δI`
    /// 2. Solve linear system: `R_loaded y = a` (SSOT complex solver)
    /// 3. Compute normalization: `denom = a^H y`
    /// 4. Normalize weights: `w = y / denom`
    ///
    /// # Parameters
    ///
    /// - `covariance`: Sample covariance matrix **R** (N×N Hermitian)
    /// - `steering`: Steering vector **a** (N×1 complex)
    ///
    /// # Returns
    ///
    /// Optimal weight vector **w** (N×1 complex) satisfying:
    /// - Unit gain constraint: w^H a = 1
    /// - Minimum output power: w^H R w is minimized
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - `covariance` is not square or empty (must be N×N with N > 0)
    /// - `steering.len() != N` (dimension mismatch)
    /// - `diagonal_loading` is invalid (NaN or negative)
    /// - Covariance matrix is singular or ill-conditioned (SSOT solver fails)
    /// - Normalization denominator is non-finite or non-positive
    ///
    /// # Mathematical Correctness
    ///
    /// The implementation guarantees:
    /// - **Unit gain**: |w^H a - 1| < ε for numerical tolerance ε
    /// - **Minimum power**: w = argmin w^H R w subject to w^H a = 1
    /// - **No silent fallbacks**: Failures are explicit errors, never disguised
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(N³) for direct solve, O(N²) for iterative
    /// - **Space Complexity**: O(N²) for covariance matrix
    /// - **Numerical Stability**: Depends on condition number and diagonal loading
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kwavers::analysis::signal_processing::beamforming::adaptive::MinimumVariance;
    /// use ndarray::{Array1, Array2};
    /// use num_complex::Complex64;
    ///
    /// let n = 8; // 8-element array
    ///
    /// // Sample covariance matrix (identity + noise)
    /// let mut cov = Array2::eye(n);
    /// for i in 0..n {
    ///     for j in 0..n {
    ///         if i != j {
    ///             cov[(i, j)] = Complex64::new(0.1, 0.0);
    ///         }
    ///     }
    /// }
    ///
    /// // Steering vector (plane wave at 0°)
    /// let steering = Array1::from_vec(vec![Complex64::new(1.0, 0.0); n]);
    ///
    /// // Compute MVDR weights
    /// let mvdr = MinimumVariance::with_diagonal_loading(1e-4);
    /// let weights = mvdr.compute_weights(&cov, &steering)?;
    ///
    /// // Verify unit gain constraint
    /// let gain: Complex64 = weights.iter()
    ///     .zip(steering.iter())
    ///     .map(|(w, a)| w.conj() * a)
    ///     .sum();
    /// assert!((gain.re - 1.0).abs() < 1e-6);
    /// assert!(gain.im.abs() < 1e-6);
    /// ```
    pub fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = covariance.nrows();

        // Validate input dimensions
        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR: covariance must be non-empty and square; got {}×{}",
                covariance.nrows(),
                covariance.ncols()
            )));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR: steering length ({}) must match covariance dimension ({})",
                steering.len(),
                n
            )));
        }
        if !self.diagonal_loading.is_finite() || self.diagonal_loading < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR: diagonal_loading must be finite and ≥ 0; got {}",
                self.diagonal_loading
            )));
        }

        // Apply diagonal loading: R_loaded = R + δI
        let mut r_loaded = covariance.clone();
        if self.diagonal_loading > 0.0 {
            let loading = Complex64::new(self.diagonal_loading, 0.0);
            for i in 0..n {
                r_loaded[(i, i)] += loading;
            }
        }

        // Solve R_loaded y = a using SSOT complex linear solver
        // This avoids explicit matrix inversion, improving numerical stability
        let y = LinearAlgebra::solve_linear_system_complex(&r_loaded, steering)?;

        // Compute normalization factor: denom = a^H y
        // For Hermitian positive-definite covariance with δ > 0, this should be real and positive
        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, y_i)| a.conj() * y_i)
            .sum();

        // Validate denominator (critical for MVDR correctness)
        let denom_re = denom.re;
        if !denom_re.is_finite() {
            return Err(KwaversError::Numerical(
                crate::domain::core::error::NumericalError::InvalidOperation(
                    "MVDR: non-finite denominator a^H R^{-1} a (covariance may be ill-conditioned)"
                        .to_string(),
                ),
            ));
        }
        if denom_re <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::domain::core::error::NumericalError::InvalidOperation(format!(
                    "MVDR: non-positive denominator a^H R^{{-1}} a = {} (covariance may not be Hermitian PD)",
                    denom_re
                )),
            ));
        }

        // Normalize to satisfy unit gain constraint: w = y / (a^H y)
        Ok(y.mapv(|x| x / denom_re))
    }

    /// Compute MVDR pseudospectrum (spatial spectrum) for a single steering vector.
    ///
    /// The MVDR pseudospectrum is defined as:
    ///
    /// ```text
    /// P_MVDR(a) = 1 / (a^H R^{-1} a)
    /// ```
    ///
    /// This is the denominator term from the weight computation.
    ///
    /// # Parameters
    ///
    /// - `covariance`: Sample covariance matrix **R** (N×N Hermitian)
    /// - `steering`: Steering vector **a** (N×1 complex)
    ///
    /// # Returns
    ///
    /// Pseudospectrum value (real, positive)
    ///
    /// # Errors
    ///
    /// Same as [`compute_weights`](Self::compute_weights).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Scan over angles to find peaks (source directions)
    /// let angles = Array1::linspace(-90.0, 90.0, 181);
    /// let mut spectrum = Vec::new();
    ///
    /// for &angle in &angles {
    ///     let steering = compute_steering_vector(n_sensors, angle);
    ///     let power = mvdr.pseudospectrum(&covariance, &steering)?;
    ///     spectrum.push(power);
    /// }
    ///
    /// // Find peaks to identify source directions
    /// ```
    pub fn pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<f64> {
        let n = covariance.nrows();

        // Validate (same as compute_weights)
        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR pseudospectrum: covariance must be square; got {}×{}",
                n,
                covariance.ncols()
            )));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(
                "MVDR pseudospectrum: steering length mismatch".to_string(),
            ));
        }

        // Apply diagonal loading
        let mut r_loaded = covariance.clone();
        if self.diagonal_loading > 0.0 {
            let loading = Complex64::new(self.diagonal_loading, 0.0);
            for i in 0..n {
                r_loaded[(i, i)] += loading;
            }
        }

        // Solve R y = a
        let y = LinearAlgebra::solve_linear_system_complex(&r_loaded, steering)?;

        // denom = a^H y = a^H R^{-1} a
        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, y_i)| a.conj() * y_i)
            .sum();

        let denom_re = denom.re;
        if !denom_re.is_finite() || denom_re <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::domain::core::error::NumericalError::InvalidOperation(
                    "MVDR pseudospectrum: invalid denominator".to_string(),
                ),
            ));
        }

        // P_MVDR = 1 / denom
        Ok(1.0 / denom_re)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::signal_processing::beamforming::test_utilities;
    use approx::assert_relative_eq;
    use ndarray::Array1;
    use std::f64::consts::PI;

    #[test]
    fn mvdr_computes_finite_weights() {
        let n = 8;
        let cov = test_utilities::create_diagonal_dominant_covariance(n, 0.1);
        let steering = test_utilities::create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::default();
        let weights = mvdr
            .compute_weights(&cov, &steering)
            .expect("weights should compute");

        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite(), "weight should be finite");
        }
    }

    #[test]
    fn mvdr_satisfies_unit_gain_constraint() {
        let n = 8;
        let cov = test_utilities::create_diagonal_dominant_covariance(n, 0.1);
        let steering = test_utilities::create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::with_diagonal_loading(1e-4);
        let weights = mvdr
            .compute_weights(&cov, &steering)
            .expect("weights should compute");

        // Check unit gain constraint: w^H a = 1
        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(&w, &a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(gain.im, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn mvdr_with_different_angles() {
        let n = 4;
        let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);

        let mvdr = MinimumVariance::default();

        for angle_deg in [-30.0, 0.0, 30.0] {
            let angle_rad = angle_deg * PI / 180.0;
            let steering = test_utilities::create_steering_vector(n, angle_rad);

            let weights = mvdr
                .compute_weights(&cov, &steering)
                .expect("weights should compute");

            // Verify unit gain
            let gain: Complex64 = weights
                .iter()
                .zip(steering.iter())
                .map(|(&w, &a)| w.conj() * a)
                .sum();

            assert_relative_eq!(gain.norm(), 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn mvdr_pseudospectrum_is_positive() {
        let n = 8;
        let cov = test_utilities::create_diagonal_dominant_covariance(n, 0.1);
        let steering = test_utilities::create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::with_diagonal_loading(1e-6);
        let spectrum = mvdr
            .pseudospectrum(&cov, &steering)
            .expect("pseudospectrum should compute");

        assert!(spectrum > 0.0);
        assert!(spectrum.is_finite());
    }

    #[test]
    fn mvdr_rejects_empty_covariance() {
        let cov = Array2::<Complex64>::zeros((0, 0));
        let steering = Array1::<Complex64>::zeros(0);

        let mvdr = MinimumVariance::default();
        let err = mvdr
            .compute_weights(&cov, &steering)
            .expect_err("should reject empty arrays");

        assert!(err.to_string().contains("non-empty"));
    }

    #[test]
    fn mvdr_rejects_non_square_covariance() {
        let cov = Array2::<Complex64>::zeros((3, 4));
        let steering = Array1::<Complex64>::zeros(3);

        let mvdr = MinimumVariance::default();
        let err = mvdr
            .compute_weights(&cov, &steering)
            .expect_err("should reject non-square covariance");

        assert!(err.to_string().contains("square"));
    }

    #[test]
    fn mvdr_rejects_dimension_mismatch() {
        let cov = test_utilities::create_test_covariance(4, 0.2, 0.1);
        let steering = Array1::zeros(5); // Wrong size

        let mvdr = MinimumVariance::default();
        let err = mvdr
            .compute_weights(&cov, &steering)
            .expect_err("should reject dimension mismatch");

        assert!(err.to_string().contains("dimension"));
    }

    #[test]
    fn mvdr_rejects_negative_diagonal_loading() {
        let cov = test_utilities::create_test_covariance(4, 0.2, 0.1);
        let steering = test_utilities::create_steering_vector(4, 0.0);

        let mvdr = MinimumVariance::with_diagonal_loading(-0.1);
        let err = mvdr
            .compute_weights(&cov, &steering)
            .expect_err("should reject negative loading");

        assert!(err.to_string().contains("≥ 0") || err.to_string().contains(">= 0"));
    }

    #[test]
    fn mvdr_rejects_nan_diagonal_loading() {
        let cov = test_utilities::create_test_covariance(4, 0.2, 0.1);
        let steering = test_utilities::create_steering_vector(4, 0.0);

        let mvdr = MinimumVariance::with_diagonal_loading(f64::NAN);
        let err = mvdr
            .compute_weights(&cov, &steering)
            .expect_err("should reject NaN loading");

        assert!(err.to_string().contains("finite"));
    }

    #[test]
    fn mvdr_default_has_small_loading() {
        let mvdr = MinimumVariance::default();
        assert_eq!(mvdr.diagonal_loading, 1e-6);
    }

    #[test]
    fn mvdr_new_has_zero_loading() {
        let mvdr = MinimumVariance::new();
        assert_eq!(mvdr.diagonal_loading, 0.0);
    }
}
