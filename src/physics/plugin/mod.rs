// src/physics/plugin/mod.rs
//! Plugin Architecture for Extensible Physics Modules
//! 
//! This module implements a plugin system that follows key design principles:
//! - **SOLID**: Single responsibility plugins with clear interfaces
//! - **CUPID**: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
//! - **GRASP**: Information expert pattern with proper encapsulation
//! - **DRY**: Shared utilities and common interfaces
//! - **KISS**: Simple, intuitive plugin API
//! - **YAGNI**: Only essential features implemented
//! - **Clean**: Clear abstractions and comprehensive documentation

pub mod adapters;
pub use adapters::{ComponentPluginAdapter, factories::{acoustic_wave_plugin, thermal_diffusion_plugin}};

#[cfg(test)]
mod tests;

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::composable::{FieldType, ValidationResult};
use ndarray::Array4;
use std::any::Any;
use std::collections::HashMap;
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

/// Core trait for physics plugins
/// 
/// This trait defines the interface that all physics plugins must implement.
/// It follows the Interface Segregation Principle by keeping the interface minimal
/// and focused on essential functionality.
pub trait PhysicsPlugin: Debug + Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;
    
    /// Get the list of field types this plugin requires as input
    fn required_fields(&self) -> Vec<FieldType>;
    
    /// Get the list of field types this plugin provides as output
    fn provided_fields(&self) -> Vec<FieldType>;
    
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
    
    /// Check if the plugin can execute with the given available fields
    fn can_execute(&self, available_fields: &[FieldType]) -> bool {
        self.required_fields()
            .iter()
            .all(|req| available_fields.contains(req))
    }
    
    /// Get current performance metrics
    fn get_metrics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
    
    /// Validate plugin state and configuration
    fn validate(&self, grid: &Grid, medium: &dyn Medium) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // Default validation checks
        if self.required_fields().is_empty() && self.provided_fields().is_empty() {
            result.add_warning("Plugin neither requires nor provides any fields".to_string());
        }
        
        result
    }
    
    /// Reset plugin state
    fn reset(&mut self) -> KwaversResult<()> {
        Ok(())
    }
    
    /// Finalize and cleanup
    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }
}

/// Context passed to plugins during update
#[derive(Debug, Clone)]
pub struct PluginContext {
    /// Current simulation step number
    pub step: usize,
    /// Total number of steps
    pub total_steps: usize,
    /// Reference frequency for the simulation
    pub frequency: f64,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

impl PluginContext {
    /// Create a new plugin context
    pub fn new(step: usize, total_steps: usize, frequency: f64) -> Self {
        Self {
            step,
            total_steps,
            frequency,
            parameters: HashMap::new(),
        }
    }
    
    /// Add a parameter to the context
    pub fn with_parameter(mut self, key: String, value: f64) -> Self {
        self.parameters.insert(key, value);
        self
    }
}

/// Plugin manager for runtime composition
pub struct PluginManager {
    /// Registered plugins
    plugins: Vec<Box<dyn PhysicsPlugin>>,
    /// Field dependency graph
    field_dependencies: HashMap<FieldType, Vec<String>>,
    /// Execution order based on dependencies
    execution_order: Vec<usize>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            field_dependencies: HashMap::new(),
            execution_order: Vec::new(),
        }
    }
    
    /// Register a plugin
    pub fn register(&mut self, plugin: Box<dyn PhysicsPlugin>) -> KwaversResult<()> {
        // Check for ID conflicts
        let new_id = plugin.metadata().id.clone();
        if self.plugins.iter().any(|p| p.metadata().id == new_id) {
            return Err(crate::error::PhysicsError::InvalidConfiguration {
                component: "PluginManager".to_string(),
                reason: format!("Plugin with ID '{}' already registered", new_id)
            }.into());
        }
        
        // Update field dependencies
        for field in plugin.provided_fields() {
            self.field_dependencies
                .entry(field)
                .or_insert_with(Vec::new)
                .push(new_id.clone());
        }
        
        self.plugins.push(plugin);
        self.compute_execution_order()?;
        
        Ok(())
    }
    
    /// Compute execution order based on dependencies
    fn compute_execution_order(&mut self) -> KwaversResult<()> {
        // Simple topological sort
        let n = self.plugins.len();
        let mut visited = vec![false; n];
        let mut order = Vec::new();
        
        fn visit(
            idx: usize,
            plugins: &[Box<dyn PhysicsPlugin>],
            visited: &mut [bool],
            order: &mut Vec<usize>,
            field_providers: &HashMap<FieldType, Vec<String>>,
        ) -> Result<(), String> {
            if visited[idx] {
                return Ok(());
            }
            
            visited[idx] = true;
            
            // Visit dependencies first
            let plugin = &plugins[idx];
            for required_field in plugin.required_fields() {
                if let Some(providers) = field_providers.get(&required_field) {
                    for provider_id in providers {
                        if let Some(provider_idx) = plugins.iter().position(|p| &p.metadata().id == provider_id) {
                            if provider_idx != idx {
                                visit(provider_idx, plugins, visited, order, field_providers)?;
                            }
                        }
                    }
                }
            }
            
            order.push(idx);
            Ok(())
        }
        
        // Build field provider map
        let mut field_providers = HashMap::new();
        for (idx, plugin) in self.plugins.iter().enumerate() {
            for field in plugin.provided_fields() {
                field_providers
                    .entry(field)
                    .or_insert_with(Vec::new)
                    .push(plugin.metadata().id.clone());
            }
        }
        
        // Visit all plugins
        for i in 0..n {
            visit(i, &self.plugins, &mut visited, &mut order, &field_providers)
                .map_err(|e| crate::error::PhysicsError::InvalidConfiguration {
                    component: "PluginManager".to_string(),
                    reason: e
                })?;
        }
        
        self.execution_order = order;
        Ok(())
    }
    
    /// Initialize all plugins
    pub fn initialize_all(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.initialize(grid, medium)?;
        }
        Ok(())
    }
    
    /// Update all plugins in dependency order
    pub fn update_all(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        for &idx in &self.execution_order {
            self.plugins[idx].update(fields, grid, medium, dt, t, context)?;
        }
        Ok(())
    }
    
    /// Get all available fields from registered plugins
    pub fn available_fields(&self) -> Vec<FieldType> {
        let mut fields = Vec::new();
        for plugin in &self.plugins {
            fields.extend(plugin.provided_fields());
        }
        fields.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        fields.dedup();
        fields
    }
    
    /// Validate all plugins
    pub fn validate_all(&self, grid: &Grid, medium: &dyn Medium) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // Check each plugin
        for plugin in &self.plugins {
            let plugin_result = plugin.validate(grid, medium);
            result.merge(plugin_result);
        }
        
        // Check for missing dependencies
        let available = self.available_fields();
        for plugin in &self.plugins {
            for required in plugin.required_fields() {
                if !available.contains(&required) {
                    result.add_error(format!(
                        "Plugin '{}' requires field '{}' which is not provided by any plugin",
                        plugin.metadata().id,
                        required.as_str()
                    ));
                }
            }
        }
        
        result
    }
    
    /// Get combined metrics from all plugins
    pub fn get_all_metrics(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut all_metrics = HashMap::new();
        for plugin in &self.plugins {
            all_metrics.insert(
                plugin.metadata().id.clone(),
                plugin.get_metrics(),
            );
        }
        all_metrics
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

// Tests moved to tests.rs
#[cfg(test)]
mod plugin_internal_tests {
    use super::*;
    
    #[derive(Debug)]
    struct TestPlugin {
        metadata: PluginMetadata,
        required: Vec<FieldType>,
        provided: Vec<FieldType>,
    }
    
    impl PhysicsPlugin for TestPlugin {
        fn metadata(&self) -> &PluginMetadata {
            &self.metadata
        }
        
        fn required_fields(&self) -> Vec<FieldType> {
            self.required.clone()
        }
        
        fn provided_fields(&self) -> Vec<FieldType> {
            self.provided.clone()
        }
        
        fn initialize(
            &mut self,
            _config: Option<Box<dyn PluginConfig>>,
            _grid: &Grid,
            _medium: &dyn Medium,
        ) -> KwaversResult<()> {
            Ok(())
        }
        
        fn update(
            &mut self,
            _fields: &mut Array4<f64>,
            _grid: &Grid,
            _medium: &dyn Medium,
            _dt: f64,
            _t: f64,
            _context: &PluginContext,
        ) -> KwaversResult<()> {
            Ok(())
        }
    }
    
    #[test]
    fn test_plugin_registration() {
        let mut manager = PluginManager::new();
        
        let plugin = Box::new(TestPlugin {
            metadata: PluginMetadata {
                id: "test_plugin".to_string(),
                name: "Test Plugin".to_string(),
                version: "1.0.0".to_string(),
                description: "A test plugin".to_string(),
                author: "Test Author".to_string(),
                license: "MIT".to_string(),
            },
            required: vec![FieldType::Pressure],
            provided: vec![FieldType::Temperature],
        });
        
        assert!(manager.register(plugin).is_ok());
        assert_eq!(manager.plugins.len(), 1);
        assert_eq!(manager.available_fields(), vec![FieldType::Temperature]);
    }
    
    #[test]
    fn test_dependency_ordering() {
        let mut manager = PluginManager::new();
        
        // Plugin A: Pressure -> Temperature
        let plugin_a = Box::new(TestPlugin {
            metadata: PluginMetadata {
                id: "plugin_a".to_string(),
                name: "Plugin A".to_string(),
                version: "1.0.0".to_string(),
                description: "Converts pressure to temperature".to_string(),
                author: "Test".to_string(),
                license: "MIT".to_string(),
            },
            required: vec![FieldType::Pressure],
            provided: vec![FieldType::Temperature],
        });
        
        // Plugin B: Temperature -> Light
        let plugin_b = Box::new(TestPlugin {
            metadata: PluginMetadata {
                id: "plugin_b".to_string(),
                name: "Plugin B".to_string(),
                version: "1.0.0".to_string(),
                description: "Converts temperature to light".to_string(),
                author: "Test".to_string(),
                license: "MIT".to_string(),
            },
            required: vec![FieldType::Temperature],
            provided: vec![FieldType::Light],
        });
        
        // Register in reverse order
        assert!(manager.register(plugin_b).is_ok());
        assert!(manager.register(plugin_a).is_ok());
        
        // Check execution order (should be A then B)
        assert_eq!(manager.execution_order.len(), 2);
        let first_plugin_id = &manager.plugins[manager.execution_order[0]].metadata().id;
        let second_plugin_id = &manager.plugins[manager.execution_order[1]].metadata().id;
        assert_eq!(first_plugin_id, "plugin_a");
        assert_eq!(second_plugin_id, "plugin_b");
    }
}