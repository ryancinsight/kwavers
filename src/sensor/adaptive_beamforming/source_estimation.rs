//! Source number estimation algorithms
//!
//! This module implements automatic source number estimation using
//! information theoretic criteria for adaptive beamforming.

use ndarray::Array2;
use num_complex::Complex64;

use super::matrix_utils::eigen_hermitian;

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

/// Estimate the number of signal sources from covariance matrix
///
/// # Arguments
/// * `covariance` - Sample covariance matrix (n x n)
/// * `num_snapshots` - Number of temporal snapshots used to compute covariance
/// * `criterion` - Information criterion to use (AIC or MDL)
///
/// # Returns
/// Estimated number of sources (0 to n-1)
///
/// # References
/// - Wax & Kailath (1985), "Detection of signals by information theoretic criteria"
pub fn estimate_num_sources(
    covariance: &Array2<Complex64>,
    num_snapshots: usize,
    criterion: SourceEstimationCriterion,
) -> usize {
    let n = covariance.nrows();
    if n == 0 || num_snapshots == 0 {
        return 0;
    }

    // Compute eigenvalues
    let (mut eigenvalues, _) = match eigen_hermitian(covariance, n) {
        Some((vals, vecs)) => (vals, vecs),
        None => return 0,
    };

    // Sort eigenvalues in descending order
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Ensure all eigenvalues are positive
    for val in &mut eigenvalues {
        if *val < 1e-12 {
            *val = 1e-12;
        }
    }

    let m = num_snapshots;
    let mut min_criterion = f64::INFINITY;
    let mut estimated_sources = 0;

    // Test each hypothesis k = 0, 1, ..., n-1
    for k in 0..(n - 1) {
        // Noise eigenvalues: λ_{k+1}, ..., λ_n
        let noise_eigs = &eigenvalues[(k + 1)..];
        let p = n - k - 1; // Number of noise eigenvalues

        if p == 0 {
            break;
        }

        // Arithmetic mean of noise eigenvalues
        let arithmetic_mean: f64 = noise_eigs.iter().sum::<f64>() / (p as f64);

        // Geometric mean of noise eigenvalues
        let log_sum: f64 = noise_eigs.iter().map(|&x| x.ln()).sum();
        let geometric_mean = (log_sum / (p as f64)).exp();

        // Avoid division by zero or invalid values
        if arithmetic_mean < 1e-12 || geometric_mean < 1e-12 {
            continue;
        }

        // Log-likelihood term
        let log_likelihood = -(m as f64) * (p as f64) * (arithmetic_mean / geometric_mean).ln();

        // Penalty term depends on criterion
        let penalty = match criterion {
            SourceEstimationCriterion::AIC => {
                // AIC penalty: 2 * number of free parameters
                // Number of parameters: k(2n - k)
                2.0 * (k as f64) * (2.0 * (n as f64) - (k as f64))
            }
            SourceEstimationCriterion::MDL => {
                // MDL penalty: 0.5 * log(m) * number of free parameters
                0.5 * (m as f64).ln() * (k as f64) * (2.0 * (n as f64) - (k as f64))
            }
        };

        let criterion_value = -log_likelihood + penalty;

        if criterion_value < min_criterion {
            min_criterion = criterion_value;
            estimated_sources = k;
        }
    }

    estimated_sources
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex64;

    #[test]
    fn test_estimate_num_sources_aic() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        let estimated = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::AIC);

        // Should return a valid estimate
        assert!(estimated < n);
    }

    #[test]
    fn test_estimate_num_sources_mdl() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        let estimated = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::MDL);

        // Should return a valid estimate
        assert!(estimated < n);
    }

    #[test]
    fn test_estimate_num_sources_mdl_conservative() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        let aic = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::AIC);
        let mdl = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::MDL);

        // MDL should be ≤ AIC (more conservative)
        assert!(mdl <= aic, "MDL ({}) should be ≤ AIC ({})", mdl, aic);
    }

    #[test]
    fn test_estimate_num_sources_high_snr() {
        let n = 6;
        let num_sources = 2;
        let num_snapshots = 200;

        // Create covariance with clear signal-noise separation
        let mut cov = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // First 2 eigenvalues are large (signals), rest are small (noise)
                    let val = if i < num_sources { 10.0 } else { 0.1 };
                    cov[(i, j)] = Complex64::new(val, 0.0);
                } else {
                    cov[(i, j)] = Complex64::new(0.01, 0.0);
                }
            }
        }

        let estimated = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::MDL);

        // Should correctly estimate close to 2 sources for high SNR
        // MDL may be conservative, so allow 1-3 sources
        assert!(
            (1..=3).contains(&estimated),
            "Should estimate 1-3 sources, got {}",
            estimated
        );
    }

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
}
