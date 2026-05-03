use crate::core::error::{KwaversError, KwaversResult};

/// Model order selection criterion
///
/// Determines the number of signal sources from eigenvalue spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelOrderCriterion {
    /// Akaike Information Criterion (AIC)
    ///
    /// Penalizes model complexity with factor 2p, where p is number of parameters.
    /// Tends to overestimate model order but detects weak signals.
    AIC,

    /// Minimum Description Length (MDL) / Bayesian Information Criterion (BIC)
    ///
    /// Penalizes model complexity with factor p·ln(N), where N is sample size.
    /// More conservative than AIC, consistent estimator as N → ∞.
    #[default]
    MDL,
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
