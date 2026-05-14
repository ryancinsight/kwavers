//! TDOA configuration types.

use super::super::config::LocalizationConfig;

/// TDOA configuration
#[derive(Debug, Clone)]
pub struct TDOAConfig {
    /// Base localization config
    pub config: LocalizationConfig,

    /// Method for time-delay estimation
    pub method: TimeDelayMethod,

    /// Number of Newton-Raphson iterations for refinement
    pub refinement_iterations: usize,

    /// Convergence tolerance for Newton-Raphson
    pub convergence_tolerance: f64,
}

/// Time-delay estimation method
#[derive(Debug, Clone, Copy)]
pub enum TimeDelayMethod {
    /// Cross-correlation at peak
    CrossCorrelation,

    /// Generalized cross-correlation (GCC)
    GeneralizedCrossCorrelation,

    /// Weighted GCC with PHAT weighting
    GCCWithPHAT,
}

impl TDOAConfig {
    /// Create new TDOA configuration
    #[must_use]
    pub fn new(config: LocalizationConfig, method: TimeDelayMethod) -> Self {
        Self {
            config,
            method,
            refinement_iterations: 5,
            convergence_tolerance: 1e-6,
        }
    }

    /// Set refinement iterations
    #[must_use]
    pub fn with_refinement_iterations(mut self, iterations: usize) -> Self {
        self.refinement_iterations = iterations;
        self
    }

    /// Set convergence tolerance
    #[must_use]
    pub fn with_convergence_tolerance(mut self, tolerance: f64) -> Self {
        self.convergence_tolerance = tolerance;
        self
    }
}

impl Default for TDOAConfig {
    fn default() -> Self {
        Self::new(
            LocalizationConfig::default(),
            TimeDelayMethod::CrossCorrelation,
        )
    }
}
