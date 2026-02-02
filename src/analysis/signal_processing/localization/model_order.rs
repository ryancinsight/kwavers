//! Model Order Selection for Source Localization
//!
//! Implements information-theoretic criteria (AIC, MDL) for automatic estimation
//! of the number of signal sources from sensor array covariance matrices.
//!
//! # Theory
//!
//! Given M sensors and K unknown sources (K < M), the spatial covariance matrix R
//! has rank K. The eigenvalues partition into:
//! - Signal subspace: K largest eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₖ (signal + noise)
//! - Noise subspace: M-K smallest eigenvalues λₖ₊₁ ≈ λₖ₊₂ ≈ ... ≈ λₘ ≈ σ²
//!
//! AIC and MDL estimate K by minimizing a penalized likelihood criterion.
//!
//! # References
//!
//! - Wax, M., & Kailath, T. (1985). "Detection of signals by information theoretic criteria"
//!   IEEE Trans. Acoust., Speech, Signal Process., 33(2), 387-392.
//! - Rissanen, J. (1978). "Modeling by shortest data description"
//!   Automatica, 14(5), 465-471.
//! - Akaike, H. (1974). "A new look at the statistical identification model"
//!   IEEE Trans. Autom. Control, 19(6), 716-723.

use crate::core::error::{KwaversError, KwaversResult};

/// Model order selection criterion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelOrderCriterion {
    /// Akaike Information Criterion (AIC)
    ///
    /// Penalizes model complexity with factor 2p, where p is number of parameters.
    /// Tends to overestimate number of sources in finite samples.
    AIC,

    /// Minimum Description Length (MDL) / Bayesian Information Criterion (BIC)
    ///
    /// Penalizes model complexity with factor p·ln(N), where N is sample size.
    /// More conservative than AIC, consistent estimator as N → ∞.
    MDL,
}

impl Default for ModelOrderCriterion {
    fn default() -> Self {
        Self::MDL // MDL is preferred in practice for its consistency
    }
}

/// Configuration for model order selection
#[derive(Debug, Clone)]
pub struct ModelOrderConfig {
    /// Information criterion to use
    pub criterion: ModelOrderCriterion,

    /// Number of sensors/channels
    pub num_sensors: usize,

    /// Number of snapshots/samples
    pub num_samples: usize,

    /// Minimum eigenvalue threshold (relative to largest eigenvalue)
    ///
    /// Eigenvalues below this threshold × λ_max are treated as numerical zero.
    /// Prevents overestimation due to numerical noise. Typical value: 1e-10.
    pub eigenvalue_threshold: f64,

    /// Maximum allowed number of sources
    ///
    /// Physical constraint: K_max < M (number of sensors).
    /// Prevents pathological estimates when noise dominates.
    pub max_sources: Option<usize>,
}

impl ModelOrderConfig {
    /// Create configuration with required parameters
    ///
    /// # Arguments
    ///
    /// * `num_sensors` - Number of sensors/array elements (M)
    /// * `num_samples` - Number of temporal snapshots (N)
    ///
    /// # Mathematical Constraints
    ///
    /// - N ≥ M (samples ≥ sensors) for non-singular covariance
    /// - M ≥ 2 (need at least 2 sensors)
    /// - Default max_sources = M - 1
    pub fn new(num_sensors: usize, num_samples: usize) -> KwaversResult<Self> {
        if num_sensors < 2 {
            return Err(KwaversError::InvalidInput(
                "Number of sensors must be ≥ 2".to_string(),
            ));
        }

        if num_samples < num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Number of samples ({}) must be ≥ number of sensors ({})",
                num_samples, num_sensors
            )));
        }

        Ok(Self {
            criterion: ModelOrderCriterion::default(),
            num_sensors,
            num_samples,
            eigenvalue_threshold: 1e-10,
            max_sources: Some(num_sensors - 1),
        })
    }

    /// Set information criterion
    pub fn with_criterion(mut self, criterion: ModelOrderCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set eigenvalue threshold
    pub fn with_eigenvalue_threshold(mut self, threshold: f64) -> Self {
        self.eigenvalue_threshold = threshold;
        self
    }

    /// Set maximum allowed sources
    pub fn with_max_sources(mut self, max_sources: usize) -> Self {
        self.max_sources = Some(max_sources);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.eigenvalue_threshold < 0.0 || self.eigenvalue_threshold > 1.0 {
            return Err(KwaversError::InvalidInput(
                "Eigenvalue threshold must be in [0, 1]".to_string(),
            ));
        }

        if let Some(max_src) = self.max_sources {
            if max_src >= self.num_sensors {
                return Err(KwaversError::InvalidInput(format!(
                    "Max sources ({}) must be < number of sensors ({})",
                    max_src, self.num_sensors
                )));
            }
        }

        Ok(())
    }
}

impl Default for ModelOrderConfig {
    fn default() -> Self {
        Self {
            criterion: ModelOrderCriterion::default(),
            num_sensors: 4,
            num_samples: 100,
            eigenvalue_threshold: 1e-10,
            max_sources: Some(3),
        }
    }
}

/// Model order selection result
#[derive(Debug, Clone)]
pub struct ModelOrderResult {
    /// Estimated number of sources
    pub num_sources: usize,

    /// Criterion values for each candidate model order k = 0, 1, ..., K_max
    pub criterion_values: Vec<f64>,

    /// Eigenvalues sorted in descending order
    pub eigenvalues: Vec<f64>,

    /// Signal subspace indices (eigenvalues 0..num_sources)
    pub signal_indices: Vec<usize>,

    /// Noise subspace indices (eigenvalues num_sources..M)
    pub noise_indices: Vec<usize>,
}

impl ModelOrderResult {
    /// Get signal subspace eigenvalues
    pub fn signal_eigenvalues(&self) -> Vec<f64> {
        self.signal_indices
            .iter()
            .map(|&i| self.eigenvalues[i])
            .collect()
    }

    /// Get noise subspace eigenvalues
    pub fn noise_eigenvalues(&self) -> Vec<f64> {
        self.noise_indices
            .iter()
            .map(|&i| self.eigenvalues[i])
            .collect()
    }

    /// Estimate noise variance (average of noise eigenvalues)
    pub fn noise_variance(&self) -> f64 {
        let noise_eigs = self.noise_eigenvalues();
        if noise_eigs.is_empty() {
            0.0
        } else {
            noise_eigs.iter().sum::<f64>() / noise_eigs.len() as f64
        }
    }
}

/// Model order estimator
#[derive(Debug)]
pub struct ModelOrderEstimator {
    config: ModelOrderConfig,
}

impl ModelOrderEstimator {
    /// Create new model order estimator
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
                crate::core::error::NumericalError::ConvergenceFailed {
                    method: "model_order_estimation".to_string(),
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
            // We want to minimize the criterion, and this term represents
            // how much the noise eigenvalues deviate from equality (g_k = a_k)
            let neg_log_likelihood =
                (n as f64) * (num_noise as f64) * (arithmetic_mean / geometric_mean).ln();

            // Number of free parameters: p(k) = k(2M - k)
            // Derivation: k source DOAs + k²/2 complex coherence matrix (Hermitian)
            let num_params = k * (2 * m - k);

            // Compute information criterion
            let criterion_value = match self.config.criterion {
                ModelOrderCriterion::AIC => {
                    // AIC(k) = 2[-log L(k)] + 2p(k)
                    2.0 * neg_log_likelihood + 2.0 * num_params as f64
                }
                ModelOrderCriterion::MDL => {
                    // MDL(k) = 2[-log L(k)] + p(k) ln(N)
                    2.0 * neg_log_likelihood + (num_params as f64) * (n as f64).ln()
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
    pub fn estimate_from_decomposition(
        &self,
        eigenvalues: &[f64],
        _eigenvectors: &ndarray::Array2<num_complex::Complex<f64>>,
    ) -> KwaversResult<ModelOrderResult> {
        self.estimate(eigenvalues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = ModelOrderConfig::new(4, 100).unwrap();
        assert_eq!(config.num_sensors, 4);
        assert_eq!(config.num_samples, 100);
        assert_eq!(config.max_sources, Some(3));
    }

    #[test]
    fn test_config_validation_too_few_sensors() {
        let result = ModelOrderConfig::new(1, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_too_few_samples() {
        let result = ModelOrderConfig::new(10, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimator_creation() {
        let config = ModelOrderConfig::new(4, 100).unwrap();
        let estimator = ModelOrderEstimator::new(config);
        assert!(estimator.is_ok());
    }

    #[test]
    fn test_single_source_clear_gap() {
        // Eigenvalues with clear signal/noise separation
        // 1 large eigenvalue (signal), 3 small eigenvalues (noise)
        let eigenvalues = vec![10.0, 1.0, 1.0, 1.0];

        let config = ModelOrderConfig::new(4, 100).unwrap();
        let estimator = ModelOrderEstimator::new(config).unwrap();

        let result = estimator.estimate(&eigenvalues).unwrap();

        // Should detect 1 source
        assert_eq!(result.num_sources, 1);
        assert_eq!(result.signal_indices, vec![0]);
        assert_eq!(result.noise_indices, vec![1, 2, 3]);
    }

    #[test]
    fn test_two_sources_clear_gap() {
        // 2 large eigenvalues (signal), 2 small eigenvalues (noise)
        let eigenvalues = vec![15.0, 10.0, 1.0, 1.0];

        let config = ModelOrderConfig::new(4, 100).unwrap();
        let estimator = ModelOrderEstimator::new(config).unwrap();

        let result = estimator.estimate(&eigenvalues).unwrap();

        // Should detect 2 sources
        assert_eq!(result.num_sources, 2);
        assert_eq!(result.signal_indices, vec![0, 1]);
        assert_eq!(result.noise_indices, vec![2, 3]);
    }

    #[test]
    fn test_no_sources_all_noise() {
        // All eigenvalues approximately equal (pure noise)
        let eigenvalues = vec![1.01, 1.0, 0.99, 1.0];

        let config = ModelOrderConfig::new(4, 100).unwrap();
        let estimator = ModelOrderEstimator::new(config).unwrap();

        let result = estimator.estimate(&eigenvalues).unwrap();

        // Should detect 0 sources (all noise)
        assert_eq!(result.num_sources, 0);
        assert_eq!(result.signal_indices.len(), 0);
        assert_eq!(result.noise_indices.len(), 4);
    }

    #[test]
    fn test_aic_vs_mdl() {
        let eigenvalues = vec![10.0, 5.0, 1.0, 1.0];

        // Test with AIC
        let config_aic = ModelOrderConfig::new(4, 100)
            .unwrap()
            .with_criterion(ModelOrderCriterion::AIC);
        let estimator_aic = ModelOrderEstimator::new(config_aic).unwrap();
        let result_aic = estimator_aic.estimate(&eigenvalues).unwrap();

        // Test with MDL
        let config_mdl = ModelOrderConfig::new(4, 100)
            .unwrap()
            .with_criterion(ModelOrderCriterion::MDL);
        let estimator_mdl = ModelOrderEstimator::new(config_mdl).unwrap();
        let result_mdl = estimator_mdl.estimate(&eigenvalues).unwrap();

        // Both should work (may give same or different answers)
        assert!(result_aic.num_sources <= 3);
        assert!(result_mdl.num_sources <= 3);

        // MDL typically more conservative (same or fewer sources than AIC)
        assert!(result_mdl.num_sources <= result_aic.num_sources);
    }

    #[test]
    fn test_noise_variance_estimation() {
        let eigenvalues = vec![20.0, 15.0, 2.0, 2.0, 2.0];

        let config = ModelOrderConfig::new(5, 100).unwrap();
        let estimator = ModelOrderEstimator::new(config).unwrap();

        let result = estimator.estimate(&eigenvalues).unwrap();

        // Noise variance should be close to 2.0
        let noise_var = result.noise_variance();
        assert!((noise_var - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_eigenvalue_threshold_filtering() {
        // Include a very small eigenvalue that should be filtered
        let eigenvalues = vec![10.0, 1.0, 1.0, 1e-12];

        let config = ModelOrderConfig::new(4, 100)
            .unwrap()
            .with_eigenvalue_threshold(1e-10);
        let estimator = ModelOrderEstimator::new(config).unwrap();

        let result = estimator.estimate(&eigenvalues).unwrap();

        // Should still work despite near-zero eigenvalue
        assert!(result.num_sources <= 3);
    }

    #[test]
    fn test_max_sources_constraint() {
        let eigenvalues = vec![10.0, 9.0, 8.0, 7.0, 6.0];

        // Limit to max 2 sources
        let config = ModelOrderConfig::new(5, 100).unwrap().with_max_sources(2);
        let estimator = ModelOrderEstimator::new(config).unwrap();

        let result = estimator.estimate(&eigenvalues).unwrap();

        // Should not exceed max_sources
        assert!(result.num_sources <= 2);
    }

    #[test]
    fn test_criterion_values_length() {
        let eigenvalues = vec![10.0, 5.0, 2.0, 1.0];

        let config = ModelOrderConfig::new(4, 100).unwrap();
        let estimator = ModelOrderEstimator::new(config).unwrap();

        let result = estimator.estimate(&eigenvalues).unwrap();

        // Should have criterion values for k = 0, 1, 2, 3 (max_sources = 3)
        assert!(result.criterion_values.len() > 0);
        assert!(result.criterion_values.len() <= 4);
    }

    #[test]
    fn test_subspace_eigenvalues() {
        let eigenvalues = vec![20.0, 15.0, 2.0, 2.0];

        let config = ModelOrderConfig::new(4, 100).unwrap();
        let estimator = ModelOrderEstimator::new(config).unwrap();

        let result = estimator.estimate(&eigenvalues).unwrap();

        let signal_eigs = result.signal_eigenvalues();
        let noise_eigs = result.noise_eigenvalues();

        // Signal eigenvalues should be larger than noise eigenvalues
        if !signal_eigs.is_empty() && !noise_eigs.is_empty() {
            let min_signal = signal_eigs.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_noise = noise_eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(min_signal >= max_noise);
        }

        // Total should equal original eigenvalues
        assert_eq!(signal_eigs.len() + noise_eigs.len(), eigenvalues.len());
    }
}
