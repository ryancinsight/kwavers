//! Solver feature enablement system
//!
//! This module provides a comprehensive feature management system that allows
//! enabling/disabling solver features at runtime for better performance and flexibility.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Solver features that can be enabled or disabled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverFeature {
    /// Enable reconstruction capabilities
    Reconstruction,
    /// Enable time reversal methods
    TimeReversal,
    /// Enable adaptive mesh refinement
    AdaptiveMeshRefinement,
    /// Enable GPU acceleration
    GpuAcceleration,
    /// Enable detailed logging
    DetailedLogging,
    /// Enable validation mode (for testing)
    ValidationMode,
    /// Enable high-precision mode
    HighPrecision,
    /// Enable multi-threaded execution
    MultiThreaded,
    /// Enable memory optimization
    MemoryOptimization,
    /// Enable experimental features
    ExperimentalFeatures,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct SolverFeatureSet: u32 {
        /// Reconstruction feature flag
        const RECONSTRUCTION = 1 << 0;
        /// Time reversal feature flag
        const TIME_REVERSAL = 1 << 1;
        /// Adaptive mesh refinement feature flag
        const ADAPTIVE_MESH_REFINEMENT = 1 << 2;
        /// GPU acceleration feature flag
        const GPU_ACCELERATION = 1 << 3;
        /// Detailed logging feature flag
        const DETAILED_LOGGING = 1 << 4;
        /// Validation mode feature flag
        const VALIDATION_MODE = 1 << 5;
        /// High precision mode feature flag
        const HIGH_PRECISION = 1 << 6;
        /// Multi-threaded execution feature flag
        const MULTI_THREADED = 1 << 7;
        /// Memory optimization feature flag
        const MEMORY_OPTIMIZATION = 1 << 8;
        /// Experimental features flag
        const EXPERIMENTAL_FEATURES = 1 << 9;
    }
}

// Manual serde implementation for bitflags 2.x compatibility
impl Serialize for SolverFeatureSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SolverFeatureSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        SolverFeatureSet::from_bits(bits).ok_or_else(|| {
            serde::de::Error::custom(format!("invalid SolverFeatureSet bits: {}", bits))
        })
    }
}

impl SolverFeatureSet {
    /// Create a new feature set with all features disabled
    pub fn new() -> Self {
        Self::empty()
    }

    /// Enable a specific feature
    pub fn enable(&mut self, feature: SolverFeature) {
        match feature {
            SolverFeature::Reconstruction => self.insert(Self::RECONSTRUCTION),
            SolverFeature::TimeReversal => self.insert(Self::TIME_REVERSAL),
            SolverFeature::AdaptiveMeshRefinement => self.insert(Self::ADAPTIVE_MESH_REFINEMENT),
            SolverFeature::GpuAcceleration => self.insert(Self::GPU_ACCELERATION),
            SolverFeature::DetailedLogging => self.insert(Self::DETAILED_LOGGING),
            SolverFeature::ValidationMode => self.insert(Self::VALIDATION_MODE),
            SolverFeature::HighPrecision => self.insert(Self::HIGH_PRECISION),
            SolverFeature::MultiThreaded => self.insert(Self::MULTI_THREADED),
            SolverFeature::MemoryOptimization => self.insert(Self::MEMORY_OPTIMIZATION),
            SolverFeature::ExperimentalFeatures => self.insert(Self::EXPERIMENTAL_FEATURES),
        }
    }

    /// Disable a specific feature
    pub fn disable(&mut self, feature: SolverFeature) {
        match feature {
            SolverFeature::Reconstruction => self.remove(Self::RECONSTRUCTION),
            SolverFeature::TimeReversal => self.remove(Self::TIME_REVERSAL),
            SolverFeature::AdaptiveMeshRefinement => self.remove(Self::ADAPTIVE_MESH_REFINEMENT),
            SolverFeature::GpuAcceleration => self.remove(Self::GPU_ACCELERATION),
            SolverFeature::DetailedLogging => self.remove(Self::DETAILED_LOGGING),
            SolverFeature::ValidationMode => self.remove(Self::VALIDATION_MODE),
            SolverFeature::HighPrecision => self.remove(Self::HIGH_PRECISION),
            SolverFeature::MultiThreaded => self.remove(Self::MULTI_THREADED),
            SolverFeature::MemoryOptimization => self.remove(Self::MEMORY_OPTIMIZATION),
            SolverFeature::ExperimentalFeatures => self.remove(Self::EXPERIMENTAL_FEATURES),
        }
    }

    /// Check if a specific feature is enabled
    pub fn is_enabled(&self, feature: SolverFeature) -> bool {
        match feature {
            SolverFeature::Reconstruction => self.contains(Self::RECONSTRUCTION),
            SolverFeature::TimeReversal => self.contains(Self::TIME_REVERSAL),
            SolverFeature::AdaptiveMeshRefinement => self.contains(Self::ADAPTIVE_MESH_REFINEMENT),
            SolverFeature::GpuAcceleration => self.contains(Self::GPU_ACCELERATION),
            SolverFeature::DetailedLogging => self.contains(Self::DETAILED_LOGGING),
            SolverFeature::ValidationMode => self.contains(Self::VALIDATION_MODE),
            SolverFeature::HighPrecision => self.contains(Self::HIGH_PRECISION),
            SolverFeature::MultiThreaded => self.contains(Self::MULTI_THREADED),
            SolverFeature::MemoryOptimization => self.contains(Self::MEMORY_OPTIMIZATION),
            SolverFeature::ExperimentalFeatures => self.contains(Self::EXPERIMENTAL_FEATURES),
        }
    }

    /// Enable all features
    pub fn enable_all(&mut self) {
        *self = Self::all();
    }

    /// Disable all features
    pub fn disable_all(&mut self) {
        *self = Self::empty();
    }

    /// Get preset configuration for accuracy-optimized simulation
    pub fn accuracy_optimized() -> Self {
        let mut features = Self::new();
        features.insert(Self::HIGH_PRECISION);
        features.insert(Self::ADAPTIVE_MESH_REFINEMENT);
        features.insert(Self::DETAILED_LOGGING);
        features
    }

    /// Get preset configuration for performance-optimized simulation
    pub fn performance_optimized() -> Self {
        let mut features = Self::new();
        features.insert(Self::GPU_ACCELERATION);
        features.insert(Self::MULTI_THREADED);
        features.insert(Self::MEMORY_OPTIMIZATION);
        features
    }

    /// Get preset configuration for debugging
    pub fn debugging() -> Self {
        let mut features = Self::new();
        features.insert(Self::DETAILED_LOGGING);
        features.insert(Self::VALIDATION_MODE);
        features
    }

    /// Convert to feature names for display
    pub fn enabled_features(&self) -> Vec<&'static str> {
        let mut features = Vec::new();

        if self.contains(Self::RECONSTRUCTION) {
            features.push("Reconstruction");
        }
        if self.contains(Self::TIME_REVERSAL) {
            features.push("Time Reversal");
        }
        if self.contains(Self::ADAPTIVE_MESH_REFINEMENT) {
            features.push("Adaptive Mesh Refinement");
        }
        if self.contains(Self::GPU_ACCELERATION) {
            features.push("GPU Acceleration");
        }
        if self.contains(Self::DETAILED_LOGGING) {
            features.push("Detailed Logging");
        }
        if self.contains(Self::VALIDATION_MODE) {
            features.push("Validation Mode");
        }
        if self.contains(Self::HIGH_PRECISION) {
            features.push("High Precision");
        }
        if self.contains(Self::MULTI_THREADED) {
            features.push("Multi-Threaded");
        }
        if self.contains(Self::MEMORY_OPTIMIZATION) {
            features.push("Memory Optimization");
        }
        if self.contains(Self::EXPERIMENTAL_FEATURES) {
            features.push("Experimental Features");
        }

        features
    }
}

impl fmt::Display for SolverFeatureSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let features = self.enabled_features();
        if features.is_empty() {
            write!(f, "No features enabled")
        } else {
            write!(f, "Features: {}", features.join(", "))
        }
    }
}

/// Feature manager that handles feature enablement and validation
#[derive(Debug, Clone)]
pub struct FeatureManager {
    features: SolverFeatureSet,
    available_features: SolverFeatureSet,
}

impl FeatureManager {
    /// Create a new feature manager with all features available
    pub fn new() -> Self {
        Self {
            features: SolverFeatureSet::new(),
            available_features: SolverFeatureSet::all(),
        }
    }

    /// Create a feature manager with only specified available features
    pub fn with_available_features(available: SolverFeatureSet) -> Self {
        Self {
            features: SolverFeatureSet::new(),
            available_features: available,
        }
    }

    /// Enable a feature if it's available
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
    pub fn is_enabled(&self, feature: SolverFeature) -> bool {
        self.features.is_enabled(feature)
    }

    /// Check if a feature is available
    pub fn is_available(&self, feature: SolverFeature) -> bool {
        self.available_features.is_enabled(feature)
    }

    /// Get the current feature set
    pub fn feature_set(&self) -> SolverFeatureSet {
        self.features
    }

    /// Get available features
    pub fn available_features(&self) -> SolverFeatureSet {
        self.available_features
    }

    /// Validate that required features are available
    pub fn validate_required_features(&self, required: &[SolverFeature]) -> Result<(), String> {
        for feature in required {
            if !self.is_available(*feature) {
                return Err(format!("Required feature {:?} is not available", feature));
            }
        }
        Ok(())
    }

    /// Get feature summary for display
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_set_operations() {
        let mut features = SolverFeatureSet::new();

        // Test enabling features
        features.enable(SolverFeature::Reconstruction);
        features.enable(SolverFeature::GpuAcceleration);

        assert!(features.is_enabled(SolverFeature::Reconstruction));
        assert!(features.is_enabled(SolverFeature::GpuAcceleration));
        assert!(!features.is_enabled(SolverFeature::TimeReversal));

        // Test disabling features
        features.disable(SolverFeature::Reconstruction);
        assert!(!features.is_enabled(SolverFeature::Reconstruction));
        assert!(features.is_enabled(SolverFeature::GpuAcceleration));
    }

    #[test]
    fn test_preset_configurations() {
        let accuracy_features = SolverFeatureSet::accuracy_optimized();
        assert!(accuracy_features.is_enabled(SolverFeature::HighPrecision));
        assert!(accuracy_features.is_enabled(SolverFeature::AdaptiveMeshRefinement));

        let performance_features = SolverFeatureSet::performance_optimized();
        assert!(performance_features.is_enabled(SolverFeature::GpuAcceleration));
        assert!(performance_features.is_enabled(SolverFeature::MultiThreaded));
    }

    #[test]
    fn test_feature_manager() {
        let mut manager = FeatureManager::new();

        // Test enabling available feature
        assert!(manager
            .enable_feature(SolverFeature::Reconstruction)
            .is_ok());
        assert!(manager.is_enabled(SolverFeature::Reconstruction));

        // Test enabling unavailable feature (should be available by default)
        assert!(manager.enable_feature(SolverFeature::TimeReversal).is_ok());

        // Test with limited available features
        let limited_features =
            SolverFeatureSet::RECONSTRUCTION | SolverFeatureSet::GPU_ACCELERATION;
        let mut limited_manager = FeatureManager::with_available_features(limited_features);

        assert!(limited_manager
            .enable_feature(SolverFeature::Reconstruction)
            .is_ok());
        assert!(limited_manager
            .enable_feature(SolverFeature::TimeReversal)
            .is_err());
    }

    #[test]
    fn test_feature_display() {
        let mut features = SolverFeatureSet::new();
        features.enable(SolverFeature::Reconstruction);
        features.enable(SolverFeature::GpuAcceleration);

        let display = format!("{}", features);
        assert!(display.contains("Reconstruction"));
        assert!(display.contains("GPU Acceleration"));
    }
}
