//! Plugin metadata and configuration
//!
//! This module defines metadata structures and configuration traits for plugins.

use crate::validation::ValidationResult;
use std::any::Any;
use std::fmt::Debug;

/// Metadata about a physics plugin
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Unique identifier for the plugin
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Version string (semver format)
    pub version: String,
    /// Brief description of functionality
    pub description: String,
    /// Author information
    pub author: String,
    /// License identifier (e.g., "MIT", "Apache-2.0")
    pub license: String,
}

/// Configuration for a physics plugin
pub trait PluginConfig: Debug + Send + Sync {
    /// Validate the configuration
    fn validate(&self) -> ValidationResult;

    /// Clone the configuration as a boxed Any for type erasure
    fn clone_boxed(&self) -> Box<dyn Any + Send + Sync>;
}
