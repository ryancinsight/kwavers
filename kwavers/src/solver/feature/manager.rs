//! [`FeatureManager`]: runtime feature availability and enablement gating.

use super::feature_set::{SolverFeature, SolverFeatureSet};

/// Feature manager that handles feature enablement and validation
#[derive(Debug, Clone)]
pub struct FeatureManager {
    features: SolverFeatureSet,
    available_features: SolverFeatureSet,
}

impl FeatureManager {
    /// Create a new feature manager with all features available
    #[must_use]
    pub fn new() -> Self {
        Self {
            features: SolverFeatureSet::new(),
            available_features: SolverFeatureSet::all(),
        }
    }

    /// Create a feature manager with only specified available features
    #[must_use]
    pub fn with_available_features(available: SolverFeatureSet) -> Self {
        Self {
            features: SolverFeatureSet::new(),
            available_features: available,
        }
    }

    /// Enable a feature if it's available
    /// # Errors
    /// Returns `Err` if `feature` is not in the available set.
    pub fn enable_feature(&mut self, feature: SolverFeature) -> Result<(), String> {
        if !self.available_features.is_enabled(feature) {
            return Err(format!("Feature {:?} is not available", feature));
        }
        self.features.enable(feature);
        Ok(())
    }

    /// Disable a feature
    pub fn disable_feature(&mut self, feature: SolverFeature) {
        self.features.disable(feature);
    }

    /// Check if a feature is enabled
    #[must_use]
    pub fn is_enabled(&self, feature: SolverFeature) -> bool {
        self.features.is_enabled(feature)
    }

    /// Check if a feature is available
    #[must_use]
    pub fn is_available(&self, feature: SolverFeature) -> bool {
        self.available_features.is_enabled(feature)
    }

    /// Get the current feature set
    #[must_use]
    pub fn feature_set(&self) -> SolverFeatureSet {
        self.features
    }

    /// Get available features
    #[must_use]
    pub fn available_features(&self) -> SolverFeatureSet {
        self.available_features
    }

    /// Validate that required features are available
    /// # Errors
    /// Returns `Err` naming the first unavailable required feature.
    pub fn validate_required_features(&self, required: &[SolverFeature]) -> Result<(), String> {
        for feature in required {
            if !self.is_available(*feature) {
                return Err(format!("Required feature {:?} is not available", feature));
            }
        }
        Ok(())
    }

    /// Get feature summary for display
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Enabled: {} | Available: {}",
            self.features, self.available_features
        )
    }
}

impl Default for FeatureManager {
    fn default() -> Self {
        Self::new()
    }
}
