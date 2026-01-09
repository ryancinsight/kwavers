//! Plugin system for acoustic simulations
//!
//! This module provides a flexible plugin architecture for extending
//! simulation capabilities without modifying core code.

pub mod execution;
// pub mod factory; // Removed
pub mod field_access;
pub mod manager;
pub mod metadata;

pub mod pam;
// pub mod seismic_imaging; // Moved to solver
pub mod transducer_field;

use crate::core::error::KwaversResult;
use crate::domain::field::indices;
// ... imports ...
use crate::domain::boundary::Boundary;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use ndarray::{Array3, Array4};
use std::any::Any;
use std::fmt::Debug;

pub use execution::{ExecutionStrategy, PluginExecutor, SequentialStrategy};
// pub use factory::PluginFactory; // Removed

pub use manager::PluginManager;
pub use metadata::PluginMetadata;

pub use pam::PAMPlugin;
// Seismic imaging moved to solver
pub use transducer_field::{TransducerFieldCalculatorPlugin, TransducerGeometry};

/// State of a plugin
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PluginState {
    /// Plugin is created but not initialized
    Created,
    /// Plugin is configured with parameters
    Configured,
    /// Plugin is initialized and ready
    Initialized,
    /// Plugin is actively processing
    Running,
    /// Plugin is paused
    Paused,
    /// Plugin encountered an error
    Error,
    /// Plugin has been finalized
    Finalized,
}

/// Priority levels for plugin execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PluginPriority {
    /// Lowest priority - executed last
    Low = 0,
    /// Normal priority - default
    Normal = 1,
    /// High priority - executed early
    High = 2,
    /// Critical priority - executed first
    Critical = 3,
}

/// Context passed to plugins during execution
#[derive(Debug)]
pub struct PluginContext<'a> {
    /// Additional fields for plugin communication
    pub extra_fields: &'a PluginFields,
    /// Acoustic sources in the simulation
    pub sources: &'a [Box<dyn Source>],
    /// Boundary conditions
    pub boundary: &'a mut dyn Boundary,
}

/// Alias for physics-specific plugins
pub type PhysicsPlugin = dyn Plugin;

/// Field accessor for plugins
pub type FieldAccessor = PluginFields;

/// Core trait that all plugins must implement
pub trait Plugin: Debug + Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Get current plugin state
    fn state(&self) -> PluginState;

    /// Set plugin state
    fn set_state(&mut self, state: PluginState);

    /// Get required fields for this plugin
    fn required_fields(&self) -> Vec<UnifiedFieldType>;

    /// Get fields provided by this plugin
    fn provided_fields(&self) -> Vec<UnifiedFieldType>;

    /// Update the plugin with current fields
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()>;

    /// Initialize the plugin
    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        Ok(())
    }

    /// Finalize the plugin
    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }

    /// Reset plugin state
    fn reset(&mut self) -> KwaversResult<()> {
        Ok(())
    }

    /// Get diagnostic information
    fn diagnostics(&self) -> String {
        format!("Plugin: {:?}", self.metadata())
    }

    /// Get stability constraints for time stepping
    fn stability_constraints(&self) -> crate::solver::time_integration::StabilityConstraints {
        // Default: no constraints
        crate::solver::time_integration::StabilityConstraints::default()
    }

    /// Get plugin priority
    fn priority(&self) -> PluginPriority {
        PluginPriority::Normal
    }

    /// Check if plugin is compatible with another plugin
    fn is_compatible_with(&self, _other: &dyn Plugin) -> bool {
        true
    }

    /// Convert to Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Convert to mutable Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Configuration for plugins
#[derive(Debug, Clone)]
pub struct PluginConfig {
    /// Plugin-specific parameters
    pub parameters: std::collections::HashMap<String, ConfigValue>,
    /// Enable debug output
    pub debug: bool,
    /// Maximum memory usage in bytes
    pub max_memory: Option<usize>,
    /// Number of threads to use
    pub num_threads: Option<usize>,
}

/// Configuration value types
#[derive(Debug, Clone)]
pub enum ConfigValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Array(Vec<f64>),
}

/// Container for fields that plugins can access and modify
#[derive(Debug)]
pub struct PluginFields {
    /// Pressure field
    pub pressure: Array3<f64>,
    /// Velocity field (optional)
    pub velocity: Option<Array3<f64>>,
    /// Temperature field (optional)
    pub temperature: Option<Array3<f64>>,
    /// Additional custom fields
    pub custom: std::collections::HashMap<String, Array3<f64>>,
}

impl PluginFields {
    /// Create new plugin fields container
    #[must_use]
    pub fn new(pressure: Array3<f64>) -> Self {
        Self {
            pressure,
            velocity: None,
            temperature: None,
            custom: std::collections::HashMap::new(),
        }
    }

    /// Add a custom field
    pub fn add_custom(&mut self, name: String, field: Array3<f64>) {
        self.custom.insert(name, field);
    }

    /// Get a custom field
    #[must_use]
    pub fn get_custom(&self, name: &str) -> Option<&Array3<f64>> {
        self.custom.get(name)
    }

    /// Get a mutable custom field
    pub fn get_custom_mut(&mut self, name: &str) -> Option<&mut Array3<f64>> {
        self.custom.get_mut(name)
    }
}
