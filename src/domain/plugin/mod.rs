//! Plugin interfaces and contracts
//!
//! This module defines the core traits and types for the plugin system,
//! allowing loose coupling between the solver orchestration and physics implementations.

pub mod field_access;
pub mod metadata;

pub use field_access::{FieldAccessor, PluginFields};
pub use metadata::PluginMetadata;

use crate::core::error::KwaversResult;
use crate::domain::boundary::Boundary;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use ndarray::Array4;
use std::any::Any;
use std::fmt::Debug;

/// State of a plugin
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    fn stability_constraints(&self) -> TransformationStabilityConstraints {
        TransformationStabilityConstraints::default()
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

// Support struct for stability - usually imported from solver but we can define basic needs here or import
// Since this is domain, it shouldn't import from solver.
// We will use a simplified struct or move StabilityConstraints to domain if needed.
// For now, let's assume we can define a simple one or use an associated type.
// But checking the original code, it imported `crate::solver::time_integration::StabilityConstraints`.
// This is a dependency inversion violation if domain -> solver.
// We should probably move StabilityConstraints to domain too or use a generic.
// For now, I'll stub it here to break the dependency chain.

#[derive(Debug, Clone, Default)]
pub struct TransformationStabilityConstraints {
    pub cfl_limit: Option<f64>,
}
