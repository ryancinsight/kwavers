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
//! ```text
//! w = R^{-1} a / (a^H R^{-1} a)
//! ```
//!
//! ## Diagonal Loading
//!
//! ```text
//! R_loaded = R + δI
//! w = R_loaded^{-1} a / (a^H R_loaded^{-1} a)
//! ```
//!
//! # Architectural Intent (SSOT)
//!
//! - **NO local matrix inversion** - uses `math::linear_algebra::LinearAlgebra::solve_linear_system_complex`
//! - **NO silent fallbacks** - returns `Err(...)` on numerical failure
//! - **NO error masking** - all failures are explicit and surfaced to caller
//! - **NO dummy weights** - never returns steering vector as disguised fallback
//!
//! # References
//!
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis." *Proc. IEEE*, 57(8).
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley. Chapter 6.
//! - Li, J., Stoica, P., & Wang, Z. (2003). "On robust Capon beamforming." *IEEE TSP*, 51(7).

use num_complex::Complex64;

/// Minimum Variance Distortionless Response (MVDR / Capon) beamformer.
///
/// Minimizes output power subject to unit gain in the look direction:
/// `w = R^{-1} a / (a^H R^{-1} a)`
#[derive(Debug, Clone)]
pub struct MinimumVariance {
    /// Diagonal loading factor δ ≥ 0 for numerical stability.
    pub diagonal_loading: f64,
}

impl Default for MinimumVariance {
    fn default() -> Self {
        Self {
            diagonal_loading: 1e-6,
        }
    }
}

impl MinimumVariance {
    /// Create MVDR beamformer with no diagonal loading.
    ///
    /// **Warning**: May fail for ill-conditioned covariance matrices.
    #[must_use]
    pub fn new() -> Self {
        Self {
            diagonal_loading: 0.0,
        }
    }

    /// Create MVDR beamformer with custom diagonal loading δ.
    #[must_use]
    pub fn with_diagonal_loading(diagonal_loading: f64) -> Self {
        Self { diagonal_loading }
    }

    /// Apply diagonal loading to a covariance clone and validate inputs.
    pub(super) fn loaded_covariance(
        &self,
        covariance: &ndarray::Array2<Complex64>,
        steering_len: usize,
    ) -> crate::core::error::KwaversResult<ndarray::Array2<Complex64>> {
        use crate::core::error::KwaversError;

        let n = covariance.nrows();

        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR: covariance must be non-empty and square; got {}×{}",
                covariance.nrows(),
                covariance.ncols()
            )));
        }
        if steering_len != n {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR: steering length ({}) must match covariance dimension ({})",
                steering_len, n
            )));
        }
        if !self.diagonal_loading.is_finite() || self.diagonal_loading < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR: diagonal_loading must be finite and ≥ 0; got {}",
                self.diagonal_loading
            )));
        }

        let mut r_loaded = covariance.clone();
        if self.diagonal_loading > 0.0 {
            let loading = Complex64::new(self.diagonal_loading, 0.0);
            for i in 0..n {
                r_loaded[(i, i)] += loading;
            }
        }

        Ok(r_loaded)
    }
}

mod spectrum;
#[cfg(test)]
mod tests;
mod weights;
