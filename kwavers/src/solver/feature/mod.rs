//! Solver feature enablement system.
//!
//! Provides runtime feature management: enable/disable solver features for
//! performance and flexibility. See [`FeatureManager`] for the primary entry point.

pub mod config;
pub mod feature_set;
pub mod manager;

#[cfg(test)]
mod tests;

pub use config::{FallbackBehavior, FeatureBasedConfig};
pub use feature_set::{SolverFeature, SolverFeatureSet};
pub use manager::FeatureManager;
