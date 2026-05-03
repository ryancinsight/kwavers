//! `SensitivityConfig` — configuration for sensitivity analysis.

/// Configuration for sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityConfig {
    /// Number of samples for Monte Carlo analysis
    pub num_samples: usize,
    /// Confidence level for sensitivity indices
    pub confidence_level: f64,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            confidence_level: 0.95,
        }
    }
}
