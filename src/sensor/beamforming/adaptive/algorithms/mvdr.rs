//! Minimum Variance Distortionless Response (MVDR/Capon) beamformer
//!
//! # SSOT / correctness stance
//! This implementation must not perform local matrix inversion or silently fall back to
//! non-MVDR weights. It uses the SSOT complex linear solver:
//! `crate::utils::linear_algebra::LinearAlgebra::solve_linear_system_complex`.
//!
//! # Mathematical definition
//! The MVDR/Capon weight vector is
//!
//! `w = R_loaded^{-1} a / (aᴴ R_loaded^{-1} a)`
//!
//! where `R_loaded = R + δ I`, with `δ >= 0`, and `a` is the steering vector.
//!
//! # Error policy (no masking)
//! - Dimension mismatches, singular systems, or invalid denominators are explicit errors.
//! - Callers must decide how to handle failure; this module will not return `steering` as a
//!   disguised fallback.

use crate::error::{KwaversError, KwaversResult};
use crate::solver::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::BeamformingAlgorithm;

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
    /// Create MVDR beamformer with custom diagonal loading.
    #[must_use]
    pub fn with_diagonal_loading(diagonal_loading: f64) -> Self {
        Self { diagonal_loading }
    }

    /// Compute MVDR weights **without requiring the caller to import the trait**.
    ///
    /// This is a convenience inherent method that forwards to the SSOT implementation of the
    /// MVDR linear solve and returns `KwaversResult`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `covariance` is not square or has `n == 0`
    /// - `steering.len() != n`
    /// - `diagonal_loading` is invalid (`NaN` or negative)
    /// - the SSOT complex linear solve fails (e.g., singular matrix)
    /// - the MVDR normalization denominator is non-finite or non-positive
    pub fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        <Self as BeamformingAlgorithm>::compute_weights(self, covariance, steering)
    }
}

impl BeamformingAlgorithm for MinimumVariance {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = covariance.nrows();
        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR: covariance must be non-empty and square; got {}x{}",
                covariance.nrows(),
                covariance.ncols()
            )));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR: steering length ({}) must match covariance dimension ({n})",
                steering.len()
            )));
        }
        if !self.diagonal_loading.is_finite() || self.diagonal_loading < 0.0 {
            return Err(KwaversError::InvalidInput(
                "MVDR: diagonal_loading must be finite and >= 0".to_string(),
            ));
        }

        // R_loaded = R + δ I
        let mut r_loaded = covariance.clone();
        if self.diagonal_loading > 0.0 {
            for i in 0..n {
                r_loaded[(i, i)] += Complex64::new(self.diagonal_loading, 0.0);
            }
        }

        // Solve R_loaded y = a (SSOT complex solve). Avoid explicit inversion.
        let y = LinearAlgebra::solve_linear_system_complex(&r_loaded, steering)?;

        // denom = aᴴ y (should be real-positive for Hermitian PD covariance with δ > 0).
        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, yi)| a.conj() * yi)
            .sum();

        let denom_re = denom.re;
        if !denom_re.is_finite() || denom_re <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::error::NumericalError::InvalidOperation(
                    "MVDR: non-positive or non-finite denominator a^H R^{-1} a".to_string(),
                ),
            ));
        }

        // w = y / (aᴴ y)
        Ok(y.mapv(|x| x / denom_re))
    }
}
