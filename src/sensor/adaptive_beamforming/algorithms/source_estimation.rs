//! Source number estimation using information theoretic criteria

use ndarray::Array2;
use num_complex::Complex64;

use super::utils::eigen_hermitian;

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
