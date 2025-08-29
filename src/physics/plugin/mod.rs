//! Plugin system for acoustic simulations
//!
//! This module provides a flexible plugin architecture for extending
//! simulation capabilities without modifying core code.

pub mod acoustic_wave_plugin;
pub mod elastic_wave_plugin;
pub mod execution;
pub mod factory;
pub mod field_access;
pub mod kzk_solver;
pub mod manager;
pub mod metadata;
pub mod mixed_domain;
pub mod seismic_imaging;
pub mod transducer_field;

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;
use std::any::Any;
use std::fmt::Debug;

pub use acoustic_wave_plugin::AcousticWavePlugin;
pub use elastic_wave_plugin::ElasticWavePlugin;
pub use execution::PluginExecutor;
pub use factory::PluginFactory;
pub use kzk_solver::{FrequencyOperator, KzkSolverPlugin};
pub use manager::PluginManager;
pub use metadata::PluginMetadata;
pub use mixed_domain::{DomainSelection, MixedDomainPropagationPlugin};
pub use seismic_imaging::{
    BoundaryType, ConvergenceCriteria, FwiParameters, ImagingCondition, MigrationAperture,
    RegularizationParameters, RtmSettings, SeismicImagingPlugin, StorageStrategy,
};
pub use transducer_field::{TransducerFieldCalculatorPlugin, TransducerGeometry};

/// State of a plugin
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PluginState {
    /// Plugin is created but not initialized
    Created,
    /// Plugin is initialized and ready
    Initialized,
    /// Plugin is actively processing
    Active,
    /// Plugin is paused
    Paused,
    /// Plugin encountered an error
    Error,
    /// Plugin is stopped
    Stopped,
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
pub type PluginContext = PluginFields;

/// Alias for physics-specific plugins
pub type PhysicsPlugin = dyn Plugin;

/// Field accessor for plugins
pub type FieldAccessor = PluginFields;

/// Core trait that all plugins must implement
pub trait Plugin: Debug + Send + Sync {
    /// Initialize the plugin with configuration
    fn initialize(&mut self, config: &PluginConfig) -> KwaversResult<()>;

    /// Execute the plugin's main functionality
    fn execute(
        &mut self,
        fields: &mut PluginFields,
        grid: &Grid,
        medium: &dyn Medium,
        time_step: f64,
    ) -> KwaversResult<()>;

    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Get current plugin state
    fn state(&self) -> PluginState;

    /// Set plugin state
    fn set_state(&mut self, state: PluginState);

    /// Get plugin priority
    fn priority(&self) -> PluginPriority {
        PluginPriority::Normal
    }

    /// Check if plugin is compatible with another plugin
    fn is_compatible_with(&self, other: &dyn Plugin) -> bool {
        true
    }

    /// Clean up resources when plugin is stopped
    fn cleanup(&mut self) -> KwaversResult<()> {
        Ok(())
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
    pub fn get_custom(&self, name: &str) -> Option<&Array3<f64>> {
        self.custom.get(name)
    }

    /// Get a mutable custom field
    pub fn get_custom_mut(&mut self, name: &str) -> Option<&mut Array3<f64>> {
        self.custom.get_mut(name)
    }
}
