//! [`FeatureBasedConfig`] and [`FallbackBehavior`] for feature-driven solver selection.

use super::feature_set::SolverFeatureSet;
use serde::{Deserialize, Serialize};

/// Configuration for feature-based solver selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBasedConfig {
    /// Features to enable for this solver
    pub enabled_features: SolverFeatureSet,
    /// Fallback behavior when features are unavailable
    pub fallback_behavior: FallbackBehavior,
}

impl Default for FeatureBasedConfig {
    fn default() -> Self {
        Self {
            enabled_features: SolverFeatureSet::new(),
            fallback_behavior: FallbackBehavior::WarnAndContinue,
        }
    }
}

/// Fallback behavior when required features are unavailable
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FallbackBehavior {
    /// Fail immediately if features are unavailable
    Fail,
    /// Warn but continue execution
    WarnAndContinue,
    /// Use alternative implementation
    UseAlternative,
    /// Disable the feature silently
    DisableSilently,
}
