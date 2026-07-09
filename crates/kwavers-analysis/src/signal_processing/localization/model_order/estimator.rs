use super::types::{ModelOrderConfig, ModelOrderCriterion, ModelOrderResult};
use kwavers_core::error::{KwaversError, KwaversResult};

/// Model order estimator
#[derive(Debug)]
pub struct ModelOrderEstimator {
    config: ModelOrderConfig,
}

impl ModelOrderEstimator {
    /// Create new model order estimator
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: ModelOrderConfig) -> KwaversResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Estimate number of sources from eigenvalues
    ///
    /// # Algorithm
    ///
    /// For each candidate model order k = 0, 1, ..., K_max:
    ///
    /// 1. Partition eigenvalues: λ₁,...,λₖ (signal), λₖ₊₁,...,λₘ (noise)
    ///
    /// 2. Compute geometric and arithmetic means of noise eigenvalues:
    ///    - Geometric: g_k = (∏ᵢ₌ₖ₊₁ᴹ λᵢ)^(1/(M-k))
    ///    - Arithmetic: a_k = (1/(M-k)) ∑ᵢ₌ₖ₊₁ᴹ λᵢ
    ///
    /// 3. Compute negative log-likelihood:
    ///    -log L(k) = N(M-k) ln(a_k / g_k)
    ///
    /// 4. Apply penalty term:
    ///    - AIC(k) = 2[-log L(k)] + 2p(k)
    ///    - MDL(k) = 2[-log L(k)] + p(k) ln(N)
    ///
    ///    where p(k) = k(2M - k) is the number of free parameters
    ///
    /// 5. Select k minimizing criterion
    ///
    /// # Arguments
    ///
    /// * `eigenvalues` - Eigenvalues sorted in descending order
    ///
    /// # Returns
    ///
    /// `ModelOrderResult` containing estimated source count and subspace partition
    ///
    /// # Mathematical Properties
    ///
    /// - AIC: Asymptotically efficient but overestimates in finite samples
    /// - MDL: Strongly consistent (converges to true order as N → ∞)
    /// - Both assume white Gaussian noise
    ///
    /// # References
    ///
    /// Wax & Kailath (1985), Equations (11)-(13)
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    ///
    pub fn estimate(&self, eigenvalues: &[f64]) -> KwaversResult<ModelOrderResult> {
        let m = self.config.num_sensors;
        let n = self.config.num_samples;

        if eigenvalues.len() != m {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} eigenvalues, got {}",
                m,
                eigenvalues.len()
            )));
        }

        // Filter out near-zero eigenvalues
        let lambda_max = eigenvalues[0];
        let threshold = self.config.eigenvalue_threshold * lambda_max;
        let valid_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .copied()
            .filter(|&lambda| lambda > threshold)
            .collect();

        if valid_eigenvalues.is_empty() {
            return Err(KwaversError::Numerical(
                kwavers_core::error::NumericalError::ConvergenceFailed {
                    method: "model_order_estimation".to_owned(),
                    iterations: 0,
                    error: threshold,
                },
            ));
        }

        // Determine search range
        let k_max = self
            .config
            .max_sources
            .unwrap_or(m - 1)
            .min(valid_eigenvalues.len() - 1);

        let mut criterion_values = Vec::with_capacity(k_max + 1);
        let mut best_k = 0;
        let mut best_criterion = f64::INFINITY;

        // Evaluate criterion for each candidate model order
        for k in 0..=k_max {
            let num_noise = m - k;

            if num_noise < 1 {
                // All eigenvalues assigned to signal subspace
                criterion_values.push(f64::INFINITY);
                continue;
            }

            // Noise subspace eigenvalues
            let noise_eigenvalues = &eigenvalues[k..m];

            // Geometric mean: g_k = (∏λᵢ)^(1/(M-k))
            let geometric_mean =
                noise_eigenvalues.iter().map(|&x| x.ln()).sum::<f64>() / num_noise as f64;
            let geometric_mean = geometric_mean.exp();

            // Arithmetic mean: a_k = (1/(M-k)) ∑λᵢ
            let arithmetic_mean = noise_eigenvalues.iter().sum::<f64>() / num_noise as f64;

            // Prevent division by zero or log of zero
            if geometric_mean <= 0.0 || arithmetic_mean <= 0.0 {
                criterion_values.push(f64::INFINITY);
                continue;
            }

            // Log-likelihood: -log L(k) = N(M-k) ln(a_k / g_k)
            let neg_log_likelihood =
                (n as f64) * (num_noise as f64) * (arithmetic_mean / geometric_mean).ln();

            // Number of free parameters: p(k) = k(2M - k)
            let num_params = k * (2 * m - k);

            // Compute information criterion
            let criterion_value = match self.config.criterion {
                ModelOrderCriterion::AIC => {
                    // AIC(k) = 2[-log L(k)] + 2p(k)
                    2.0f64.mul_add(neg_log_likelihood, 2.0 * num_params as f64)
                }
                ModelOrderCriterion::MDL => {
                    // MDL(k) = 2[-log L(k)] + p(k) ln(N)
                    2.0f64.mul_add(neg_log_likelihood, (num_params as f64) * (n as f64).ln())
                }
            };

            criterion_values.push(criterion_value);

            // Track minimum
            if criterion_value < best_criterion {
                best_criterion = criterion_value;
                best_k = k;
            }
        }

        // Build result
        let signal_indices: Vec<usize> = (0..best_k).collect();
        let noise_indices: Vec<usize> = (best_k..m).collect();

        Ok(ModelOrderResult {
            num_sources: best_k,
            criterion_values,
            eigenvalues: eigenvalues.to_vec(),
            signal_indices,
            noise_indices,
        })
    }

    /// Estimate from covariance matrix eigendecomposition
    ///
    /// Convenience method that accepts both eigenvalues and eigenvectors.
    /// Only eigenvalues are used for model order selection.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn estimate_from_decomposition(
        &self,
        eigenvalues: &[f64],
        _eigenvectors: &leto::Array2<num_complex::Complex<f64>>,
    ) -> KwaversResult<ModelOrderResult> {
        self.estimate(eigenvalues)
    }
}
