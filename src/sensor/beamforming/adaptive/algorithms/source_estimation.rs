//! Source number estimation using information theoretic criteria
//!
//! # SSOT / correctness stance (strict)
//! Classical model-order selection (Wax–Kailath AIC/MDL) requires the eigenvalues of the
//! **Hermitian complex covariance matrix**.
//!
//! Under strict SSOT, we compute eigenvalues via SSOT numerics only:
//! `crate::utils::linear_algebra::LinearAlgebra::hermitian_eigendecomposition_complex`.
//!
//! # Mathematical definition (Wax–Kailath, 1985)
//! Let `M` be the number of sensors (matrix dimension) and `N` be the number of snapshots.
//! Let `λ_1 ≥ ... ≥ λ_M` be the (real, non-negative) eigenvalues of the covariance matrix.
//! For a candidate model order `k` (number of sources), define on the noise subspace:
//! - Arithmetic mean: `a_k = (1/(M-k)) * Σ_{i=k+1..M} λ_i`
//! - Geometric mean:  `g_k = (Π_{i=k+1..M} λ_i)^(1/(M-k))`
//! Then the log-likelihood ratio term is:
//!   `L(k) = -N (M-k) ln(g_k / a_k)`
//! and the criteria are:
//! - AIC(k) = 2 L(k) + 2 k (2M - k)
//! - MDL(k) = L(k) + 0.5 k (2M - k) ln(N)
//!
//! We return `argmin_k criterion(k)` over `k ∈ [0, M-1]`.
//!
//! # Error policy (no masking)
//! - Dimension mismatches and non-finite inputs are errors.
//! - `N == 0` is an error.
//! - If eigenvalues contain non-finite values, or if arithmetic mean is non-positive, we error.
//! - We clamp eigenvalues below `floor = relative_floor * λ_max` to avoid `ln(0)` while
//!   keeping the clamp explicit and controlled.

use crate::error::{KwaversError, KwaversResult};
use crate::solver::linear_algebra::LinearAlgebra;
use ndarray::Array2;
use num_complex::Complex64;

/// Automatic source number estimation using information theoretic criteria
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
/// # SSOT behavior
/// Uses SSOT complex Hermitian eigendecomposition to obtain eigenvalues, then applies
/// Wax–Kailath AIC/MDL model-order selection.
///
/// # Numerical invariants assumed/validated
/// - `covariance` must be square with `M>0`.
/// - `num_snapshots = N > 0`.
/// - The covariance should be Hermitian PSD in theory; numerically we require eigenvalues to be
///   real (guaranteed by SSOT Hermitian eigensolver) and finite. Small negative eigenvalues due
///   to rounding are treated as an error under strict SSOT (do not mask invalid inputs).
///
/// # Errors
/// Returns an error if:
/// - input shapes are invalid
/// - `num_snapshots == 0`
/// - SSOT eigendecomposition fails
/// - any eigenvalue is non-finite
/// - the criterion computation encounters invalid logs (e.g., non-positive arithmetic mean)
pub fn estimate_num_sources(
    covariance: &Array2<Complex64>,
    num_snapshots: usize,
    criterion: SourceEstimationCriterion,
) -> KwaversResult<usize> {
    let m = covariance.nrows();
    if m == 0 || covariance.ncols() != m {
        return Err(KwaversError::InvalidInput(
            "estimate_num_sources: covariance must be a non-empty square matrix".to_string(),
        ));
    }
    if num_snapshots == 0 {
        return Err(KwaversError::InvalidInput(
            "estimate_num_sources: num_snapshots must be > 0".to_string(),
        ));
    }

    // SSOT eigendecomposition operates on Complex<f64>.
    let cov: Array2<num_complex::Complex<f64>> =
        covariance.mapv(|z| num_complex::Complex::<f64>::new(z.re, z.im));

    let (evals, _evecs) = LinearAlgebra::hermitian_eigendecomposition_complex(&cov)?;

    // Sort eigenvalues descending (λ1 >= ... >= λM).
    let mut lambda: Vec<f64> = evals.iter().copied().collect();
    for (i, &x) in lambda.iter().enumerate() {
        if !x.is_finite() {
            return Err(KwaversError::Numerical(crate::error::NumericalError::NaN {
                operation: "estimate_num_sources: eigenvalues".to_string(),
                inputs: format!("non-finite eigenvalue at index {i}"),
            }));
        }
        if x < 0.0 {
            return Err(KwaversError::Numerical(
                crate::error::NumericalError::InvalidOperation(
                    "estimate_num_sources: negative eigenvalue; covariance is not PSD under strict SSOT"
                        .to_string(),
                ),
            ));
        }
    }
    lambda.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let lambda_max = lambda[0].max(0.0);
    let floor = (1e-15f64).max(1e-12 * lambda_max);

    // Criterion evaluation over k = 0..M-1.
    let n = num_snapshots as f64;
    let ln_n = (num_snapshots as f64).ln();

    let mut best_k = 0usize;
    let mut best_score = f64::INFINITY;

    for k in 0..m {
        let noise_dim = m - k;
        if noise_dim == 0 {
            // k == M => invalid; our loop is 0..M so this doesn't occur
            continue;
        }

        // Noise eigenvalues are indices k..M-1 in descending order.
        // Clamp to explicit floor for log-safety.
        let mut sum = 0.0f64;
        let mut sum_log = 0.0f64;

        for &x in &lambda[k..m] {
            let x_clamped = x.max(floor);
            sum += x_clamped;
            sum_log += x_clamped.ln();
        }

        let a_k = sum / (noise_dim as f64);
        if !(a_k.is_finite()) || a_k <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::error::NumericalError::InvalidOperation(
                    "estimate_num_sources: non-positive or non-finite arithmetic mean of noise eigenvalues"
                        .to_string(),
                ),
            ));
        }

        // g_k = exp( (1/(M-k)) * Σ ln(λ_i) )
        let g_k = (sum_log / (noise_dim as f64)).exp();
        if !(g_k.is_finite()) || g_k <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::error::NumericalError::InvalidOperation(
                    "estimate_num_sources: non-positive or non-finite geometric mean of noise eigenvalues"
                        .to_string(),
                ),
            ));
        }

        // L(k) = -N (M-k) ln(g_k / a_k)
        let ratio = g_k / a_k;
        if !(ratio.is_finite()) || ratio <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::error::NumericalError::InvalidOperation(
                    "estimate_num_sources: invalid g_k / a_k ratio".to_string(),
                ),
            ));
        }
        let l_k = -n * (noise_dim as f64) * ratio.ln();

        // Penalty term: k(2M-k)
        let penalty = (k as f64) * (2.0 * (m as f64) - (k as f64));

        let score = match criterion {
            SourceEstimationCriterion::AIC => 2.0 * l_k + 2.0 * penalty,
            SourceEstimationCriterion::MDL => l_k + 0.5 * penalty * ln_n,
        };

        if !score.is_finite() {
            return Err(KwaversError::Numerical(crate::error::NumericalError::NaN {
                operation: "estimate_num_sources: criterion".to_string(),
                inputs: "non-finite score".to_string(),
            }));
        }

        if score < best_score {
            best_score = score;
            best_k = k;
        }
    }

    Ok(best_k)
}
