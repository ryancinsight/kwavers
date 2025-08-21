// src/physics/plugin/mod.rs
//! Plugin architecture for extensible physics
//! 
//! This module provides a flexible plugin system for physics simulations

pub mod metadata;
pub mod factory;
pub mod execution;
pub mod manager;
pub mod field_access;
pub mod acoustic_wave_plugin;
pub mod acoustic_simulation_plugins;

// Re-export core types
pub use metadata::{PluginMetadata, PluginConfig, PluginState};
pub use factory::{PluginFactory, TypedPluginFactory, PluginRegistry};
pub use execution::{ExecutionStrategy, SequentialStrategy};
pub use manager::PluginManager;
pub use field_access::{PluginFieldAccess, DirectPluginFieldAccess};

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use ndarray::Array4;
use std::collections::HashMap;
use std::fmt::Debug;

/// Context passed to plugins during execution
#[derive(Debug, Clone)]
pub struct PluginContext {
    /// Simulation metadata
    pub metadata: HashMap<String, String>,
    /// Current simulation step
    pub step: usize,
    /// Total number of steps
    pub total_steps: usize,
}

impl PluginContext {
    /// Create a new plugin context
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
            step: 0,
            total_steps: 0,
        }
    }
    
    /// Update the current step
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }
    
    /// Set total steps
    pub fn set_total_steps(&mut self, total: usize) {
        self.total_steps = total;
    }
}

/// Core plugin trait with lifecycle management
pub trait PhysicsPlugin: Debug + Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;
    /// Get stability constraints for time stepping
    fn stability_constraints(&self) -> f64 {
        1.0 // Default stable timestep multiplier
    }
    
    /// Get current plugin state
    fn state(&self) -> PluginState;
    
    /// Get the list of field types this plugin requires as input
    fn required_fields(&self) -> Vec<UnifiedFieldType>;
    
    /// Get the list of field types this plugin provides as output
    fn provided_fields(&self) -> Vec<UnifiedFieldType>;
    
    /// Initialize the plugin
    /// 
    /// This method is called once before the simulation starts.
    /// Plugins should perform any necessary setup here.
    fn initialize(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()>;
    
    /// Update the physics for one time step
    /// 
    /// This is the main computational method where the plugin performs its physics calculations.
    /// The plugin should read from required fields and write to provided fields.
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()>;
    
    /// Finalize the plugin
    /// 
    /// This method is called once after the simulation ends.
    /// Plugins should perform any necessary cleanup here.
    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }
    
    /// Get plugin-specific diagnostics
    fn diagnostics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
    
    /// Reset plugin state
    fn reset(&mut self) -> KwaversResult<()> {
        Ok(())
    }
}

/// Plugin handle for type-safe access
pub struct PluginHandle<T: PhysicsPlugin + 'static> {
    plugin: Box<T>,
    id: String,
}

impl<T: PhysicsPlugin + 'static> PluginHandle<T> {
    /// Create a new plugin handle
    pub fn new(plugin: T) -> Self {
        let id = plugin.metadata().id.clone();
        Self {
            plugin: Box::new(plugin),
            id,
        }
    }
    
    /// Get reference to the plugin
    pub fn plugin(&self) -> &T {
        &self.plugin
    }
    
    /// Get mutable reference to the plugin
    pub fn plugin_mut(&mut self) -> &mut T {
        &mut self.plugin
    }
    
    /// Get the plugin ID
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Convert to dynamic plugin
    pub fn into_dynamic(self) -> Box<dyn PhysicsPlugin> {
        self.plugin
    }
}

/// Composite plugin that combines multiple plugins
pub struct CompositePlugin {
    metadata: PluginMetadata,
    state: PluginState,
    plugins: Vec<Box<dyn PhysicsPlugin>>,
}

impl CompositePlugin {
    /// Create a new composite plugin
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            state: PluginState::Created,
            plugins: Vec::new(),
        }
    }
    
    /// Add a plugin to the composite
    pub fn add_plugin(&mut self, plugin: Box<dyn PhysicsPlugin>) {
        self.plugins.push(plugin);
    }
    
    /// Get the number of sub-plugins
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }
    
    /// Get a reference to a sub-plugin
    pub fn get_plugin(&self, index: usize) -> Option<&dyn PhysicsPlugin> {
        self.plugins.get(index).map(|p| p.as_ref())
    }
}

impl Debug for CompositePlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositePlugin")
            .field("metadata", &self.metadata)
            .field("state", &self.state)
            .field("plugin_count", &self.plugins.len())
            .finish()
    }
}

impl PhysicsPlugin for CompositePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        self.state
    }
    
    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        let mut fields = Vec::new();
        let mut provided = std::collections::HashSet::new();
        
        // Collect all provided fields
        for plugin in &self.plugins {
            for field in plugin.provided_fields() {
                provided.insert(field);
            }
        }
        
        // Collect required fields that aren't provided internally
        for plugin in &self.plugins {
            for field in plugin.required_fields() {
                if !provided.contains(&field) && !fields.contains(&field) {
                    fields.push(field);
                }
            }
        }
        
        fields
    }
    
    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        let mut fields = Vec::new();
        for plugin in &self.plugins {
            for field in plugin.provided_fields() {
                if !fields.contains(&field) {
                    fields.push(field);
                }
            }
        }
        fields
    }
    
    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.initialize(grid, medium)?;
        }
        self.state = PluginState::Initialized;
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        self.state = PluginState::Running;
        for plugin in &mut self.plugins {
            plugin.update(fields, grid, medium, dt, t, context)?;
        }
        Ok(())
    }
    
    fn finalize(&mut self) -> KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.finalize()?;
        }
        self.state = PluginState::Finalized;
        Ok(())
    }
    
    fn diagnostics(&self) -> HashMap<String, f64> {
        let mut diagnostics = HashMap::new();
        for (i, plugin) in self.plugins.iter().enumerate() {
            for (key, value) in plugin.diagnostics() {
                diagnostics.insert(format!("{}_{}", i, key), value);
            }
        }
        diagnostics
    }
}

/// Visitor pattern for plugin inspection
pub trait PluginVisitor {
    /// Visit a plugin
    fn visit(&mut self, plugin: &dyn PhysicsPlugin);
    
    /// Visit all plugins in a manager
    fn visit_all(&mut self, plugins: &[Box<dyn PhysicsPlugin>]) {
        for plugin in plugins {
            self.visit(plugin.as_ref());
        }
    }
}

/// Metadata collector visitor
pub struct MetadataCollector {
    metadata: Vec<PluginMetadata>,
}

impl MetadataCollector {
    /// Create a new metadata collector
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
        }
    }
    
    /// Get collected metadata
    pub fn metadata(&self) -> &[PluginMetadata] {
        &self.metadata
    }
}

impl PluginVisitor for MetadataCollector {
    fn visit(&mut self, plugin: &dyn PhysicsPlugin) {
        self.metadata.push(plugin.metadata().clone());
    }
}