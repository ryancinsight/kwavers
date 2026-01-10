//! Source number estimation algorithms
//!
//! This module implements automatic source number estimation using information theoretic criteria
//! for adaptive beamforming.
//!
//! # SSOT / correctness stance (strict)
//! Classical model-order selection (Wax–Kailath AIC/MDL) requires the eigenvalues of a **complex
//! Hermitian** covariance matrix.
//!
//! SSOT currently does **not** provide complex Hermitian eigendecomposition. Under strict SSOT
//! rules we therefore refuse to produce a masked estimate and instead return an explicit error.
//!
//! This prevents “working but incorrect” behavior (e.g. returning `0` sources due to a numerical
//! routine not being available).

use crate::domain::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use num_complex::Complex64;

/// Automatic source number estimation using information theoretic criteria.
///
/// Estimates the number of signal sources present using the covariance matrix
/// eigenvalues and information theoretic criteria (AIC, MDL).
///
/// # References
/// - Wax & Kailath (1985), "Detection of signals by information theoretic criteria",
///   IEEE Transactions on Acoustics, Speech, and Signal Processing, 33(2), 387-392
/// - Zhao et al. (1986), "Asymptotic equivalence of certain methods for model order
///   estimation", IEEE Transactions on Automatic Control, 31(1), 41-47
#[derive(Debug, Clone, Copy)]
pub enum SourceEstimationCriterion {
    /// Akaike Information Criterion (AIC)
    /// More liberal - may overestimate number of sources
    AIC,
    /// Minimum Description Length (MDL) / Bayesian Information Criterion (BIC)
    /// More conservative - consistent estimator
    MDL,
}

/// Estimate the number of signal sources from a complex covariance matrix.
///
/// # Strict SSOT behavior
/// This function cannot run until SSOT provides **complex Hermitian eigendecomposition**.
/// In strict SSOT mode, we return an explicit error instead of a masked default.
///
/// # Arguments
/// * `covariance` - Sample covariance matrix (n x n)
/// * `num_snapshots` - Number of temporal snapshots used to compute covariance
/// * `criterion` - Information criterion to use (AIC or MDL)
///
/// # Errors
/// Returns an error if invariants fail or if SSOT cannot provide complex eigendecomposition.
///
/// # References
/// - Wax & Kailath (1985), "Detection of signals by information theoretic criteria"
pub fn estimate_num_sources(
    covariance: &Array2<Complex64>,
    num_snapshots: usize,
    criterion: SourceEstimationCriterion,
) -> KwaversResult<usize> {
    let n = covariance.nrows();
    if n == 0 || covariance.ncols() != n {
        return Err(KwaversError::InvalidInput(
            "estimate_num_sources: covariance must be a non-empty square matrix".to_string(),
        ));
    }
    if num_snapshots == 0 {
        return Err(KwaversError::InvalidInput(
            "estimate_num_sources: num_snapshots must be > 0".to_string(),
        ));
    }

    let _ = criterion; // retained for API stability

    Err(KwaversError::Numerical(
        crate::domain::core::error::NumericalError::UnsupportedOperation {
            operation: "estimate_num_sources (complex Hermitian eigendecomposition)".to_string(),
            reason: "SSOT complex eigendecomposition is not implemented; implement it in crate::utils::linear_algebra and route source model-order selection through it".to_string(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex64;

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

    #[test]
    fn test_estimate_num_sources_requires_ssot_complex_eig() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        let err = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::AIC)
            .expect_err("strict SSOT must error without complex eigendecomposition");

        let msg = format!("{err:?}");
        assert!(
            msg.contains("UnsupportedOperation"),
            "expected UnsupportedOperation; got: {msg}"
        );
    }
}
