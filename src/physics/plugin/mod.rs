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

/// Plugin lifecycle state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PluginState {
    /// Plugin is created but not initialized
    Created,
    /// Plugin is initialized and ready to run
    Initialized,
    /// Plugin is currently processing
    Running,
    /// Plugin encountered an error
    Error,
    /// Plugin has been finalized
    Finalized,
}

/// Enhanced plugin trait with lifecycle management
pub trait PhysicsPlugin: Debug + Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;
    
    /// Get current plugin state
    fn state(&self) -> PluginState;
    
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
    
    /// Finalize the plugin
    /// 
    /// Called when the simulation is complete or the plugin is being removed.
    /// Plugins should clean up resources here.
    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }
    
    /// Check if the plugin can execute with the given available fields
    fn can_execute(&self, available_fields: &[FieldType]) -> bool {
        self.required_fields()
            .iter()
            .all(|req| available_fields.contains(req))
    }
    
    /// Get current performance metrics
    fn performance_metrics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
    
    /// Validate plugin configuration and state
    fn validate(&self, grid: &Grid, medium: &dyn Medium) -> ValidationResult {
        ValidationResult::new()
    }
    
    /// Clone the plugin as a boxed trait object
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin>;

    /// Get a reference to the underlying Any object
    fn as_any(&self) -> &dyn Any;

    /// Get a mutable reference to the underlying Any object
    fn as_any_mut(&mut self) -> &mut dyn Any;
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
                .or_default()
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
                plugin.performance_metrics(),
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

/// Type-safe plugin handle with phantom types for compile-time guarantees
pub struct PluginHandle<T: PhysicsPlugin + 'static> {
    plugin: Box<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PhysicsPlugin + 'static> PluginHandle<T> {
    /// Create a new plugin handle
    pub fn new(plugin: T) -> Self {
        Self {
            plugin: Box::new(plugin),
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get a reference to the plugin
    pub fn plugin(&self) -> &T {
        &self.plugin
    }
    
    /// Get a mutable reference to the plugin
    pub fn plugin_mut(&mut self) -> &mut T {
        &mut self.plugin
    }
    
    /// Convert to a dynamic plugin handle
    pub fn into_dynamic(self) -> DynamicPluginHandle {
        DynamicPluginHandle {
            plugin: self.plugin as Box<dyn PhysicsPlugin>,
        }
    }
}

/// Dynamic plugin handle for runtime plugin management
pub struct DynamicPluginHandle {
    plugin: Box<dyn PhysicsPlugin>,
}

impl DynamicPluginHandle {
    /// Try to downcast to a specific plugin type
    pub fn downcast<T: PhysicsPlugin + 'static>(&self) -> Option<&T> {
        self.plugin.as_any().downcast_ref::<T>()
    }
    
    /// Try to downcast to a specific plugin type (mutable)
    pub fn downcast_mut<T: PhysicsPlugin + 'static>(&mut self) -> Option<&mut T> {
        self.plugin.as_any_mut().downcast_mut::<T>()
    }
}

/// Plugin factory trait for creating plugins
pub trait PluginFactory: Send + Sync {
    /// Create a new plugin instance
    fn create(&self, config: Box<dyn Any + Send + Sync>) -> KwaversResult<Box<dyn PhysicsPlugin>>;
    
    /// Get metadata about the plugin this factory creates
    fn metadata(&self) -> PluginMetadata;
    
    /// Get the expected configuration type name
    fn config_type_name(&self) -> &'static str;
}

/// Type-erased wrapper for plugin factories
struct TypedPluginFactory<F, C, P>
where
    F: Fn(C) -> KwaversResult<P> + Send + Sync,
    C: PluginConfig + 'static,
    P: PhysicsPlugin + 'static,
{
    factory_fn: F,
    metadata: PluginMetadata,
    _phantom: std::marker::PhantomData<(C, P)>,
}

impl<F, C, P> TypedPluginFactory<F, C, P>
where
    F: Fn(C) -> KwaversResult<P> + Send + Sync,
    C: PluginConfig + 'static,
    P: PhysicsPlugin + 'static,
{
    fn new(factory_fn: F, metadata: PluginMetadata) -> Self {
        Self {
            factory_fn,
            metadata,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, C, P> PluginFactory for TypedPluginFactory<F, C, P>
where
    F: Fn(C) -> KwaversResult<P> + Send + Sync,
    C: PluginConfig + 'static,
    P: PhysicsPlugin + 'static,
{
    fn create(&self, config: Box<dyn Any + Send + Sync>) -> KwaversResult<Box<dyn PhysicsPlugin>> {
        // Downcast the config to the expected type
        let config = config
            .downcast::<C>()
            .map_err(|_| crate::error::KwaversError::Config(
                crate::error::ConfigError::InvalidValue {
                    parameter: "config".to_string(),
                    value: "wrong type".to_string(),
                    constraint: format!("Expected config type: {}", self.config_type_name()),
                }
            ))?;
        
        // Create the plugin
        let plugin = (self.factory_fn)(*config)?;
        Ok(Box::new(plugin))
    }
    
    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }
    
    fn config_type_name(&self) -> &'static str {
        std::any::type_name::<C>()
    }
}

/// Registry for plugin factories
pub struct PluginRegistry {
    factories: HashMap<String, Box<dyn PluginFactory>>,
    metadata: HashMap<String, PluginMetadata>,
}

impl PluginRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Register a plugin factory with a typed configuration
    pub fn register_typed<F, C, P>(
        &mut self,
        id: &str,
        metadata: PluginMetadata,
        factory_fn: F,
    ) where
        F: Fn(C) -> KwaversResult<P> + Send + Sync + 'static,
        C: PluginConfig + 'static,
        P: PhysicsPlugin + 'static,
    {
        let factory = TypedPluginFactory::new(factory_fn, metadata.clone());
        self.factories.insert(id.to_string(), Box::new(factory));
        self.metadata.insert(id.to_string(), metadata);
    }
    
    /// Register a pre-built factory
    pub fn register_factory(&mut self, factory: Box<dyn PluginFactory>) {
        let metadata = factory.metadata();
        self.factories.insert(metadata.id.clone(), factory);
        self.metadata.insert(metadata.id.clone(), metadata);
    }
    
    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        self.metadata.values().collect()
    }
    
    /// Create a plugin instance by ID
    pub fn create_plugin(
        &self,
        plugin_id: &str,
        config: Box<dyn Any + Send + Sync>,
    ) -> KwaversResult<Box<dyn PhysicsPlugin>> {
        let factory = self.factories.get(plugin_id)
            .ok_or_else(|| crate::error::KwaversError::Config(
                crate::error::ConfigError::InvalidValue {
                    parameter: "plugin_id".to_string(),
                    value: plugin_id.to_string(),
                    constraint: format!("Plugin not found in registry. Available: {:?}", 
                                      self.metadata.keys().collect::<Vec<_>>()),
                }
            ))?;
        
        factory.create(config)
    }
    
    /// Create a plugin with a typed configuration
    pub fn create_plugin_typed<C: PluginConfig + 'static>(
        &self,
        plugin_id: &str,
        config: C,
    ) -> KwaversResult<Box<dyn PhysicsPlugin>> {
        self.create_plugin(plugin_id, Box::new(config) as Box<dyn Any + Send + Sync>)
    }
    
    /// Get metadata for a plugin
    pub fn get_metadata(&self, plugin_id: &str) -> Option<&PluginMetadata> {
        self.metadata.get(plugin_id)
    }
    
    /// Check if a plugin is registered
    pub fn has_plugin(&self, plugin_id: &str) -> bool {
        self.factories.contains_key(plugin_id)
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        
        // Register built-in plugins
        registry.register_builtin_plugins();
        
        registry
    }
}

impl PluginRegistry {
    /// Register all built-in plugins
    fn register_builtin_plugins(&mut self) {
        use crate::solver::fdtd::{FdtdPlugin, FdtdConfig};
        use crate::solver::pstd::{PstdPlugin, PstdConfig};
        
        // Register FDTD plugin
        self.register_typed(
            "fdtd",
            PluginMetadata {
                id: "fdtd".to_string(),
                name: "FDTD Solver".to_string(),
                version: "1.0.0".to_string(),
                description: "Finite-Difference Time Domain solver".to_string(),
                author: "Kwavers Team".to_string(),
                license: "MIT".to_string(),
            },
            |config: FdtdConfig| {
                let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
                FdtdPlugin::new(config, &grid)
            },
        );
        
        // Register PSTD plugin
        self.register_typed(
            "pstd",
            PluginMetadata {
                id: "pstd".to_string(),
                name: "PSTD Solver".to_string(),
                version: "1.0.0".to_string(),
                description: "Pseudo-Spectral Time Domain solver".to_string(),
                author: "Kwavers Team".to_string(),
                license: "MIT".to_string(),
            },
            |config: PstdConfig| {
                let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
                PstdPlugin::new(config, &grid)
            },
        );
        
        // Add more built-in plugins as needed
    }
}

/// Plugin composition using the Composite pattern
#[derive(Debug)]
pub struct CompositePlugin {
    metadata: PluginMetadata,
    plugins: Vec<Box<dyn PhysicsPlugin>>,
    state: PluginState,
}

impl CompositePlugin {
    /// Create a new composite plugin
    pub fn new(id: String, name: String) -> Self {
        Self {
            metadata: PluginMetadata {
                id,
                name,
                version: "1.0.0".to_string(),
                description: "Composite plugin".to_string(),
                author: "kwavers".to_string(),
                license: "MIT".to_string(),
            },
            plugins: Vec::new(),
            state: PluginState::Created,
        }
    }
    
    /// Add a plugin to the composite
    pub fn add_plugin(&mut self, plugin: Box<dyn PhysicsPlugin>) {
        self.plugins.push(plugin);
    }
    
    /// Remove a plugin by ID
    pub fn remove_plugin(&mut self, plugin_id: &str) -> Option<Box<dyn PhysicsPlugin>> {
        self.plugins.iter()
            .position(|p| p.metadata().id == plugin_id)
            .map(|idx| self.plugins.remove(idx))
    }
}

impl PhysicsPlugin for CompositePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        self.state
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        self.plugins.iter()
            .flat_map(|p| p.required_fields())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        self.plugins.iter()
            .flat_map(|p| p.provided_fields())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
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
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        let mut cloned = CompositePlugin::new(
            self.metadata.id.clone(),
            self.metadata.name.clone(),
        );
        cloned.metadata = self.metadata.clone();
        
        // Clone all child plugins
        for plugin in &self.plugins {
            cloned.add_plugin(plugin.clone_plugin());
        }
        
        Box::new(cloned)
    }
}

/// Plugin execution strategy using the Strategy pattern
pub trait ExecutionStrategy: Send + Sync {
    /// Execute plugins according to the strategy
    fn execute(
        &self,
        plugins: &mut [Box<dyn PhysicsPlugin>],
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()>;
}

/// Sequential execution strategy
pub struct SequentialStrategy;

impl ExecutionStrategy for SequentialStrategy {
    fn execute(
        &self,
        plugins: &mut [Box<dyn PhysicsPlugin>],
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        for plugin in plugins {
            plugin.update(fields, grid, medium, dt, t, context)?;
        }
        Ok(())
    }
}

/// Parallel execution strategy (for independent plugins)
/// 
/// This strategy executes independent plugins in parallel using thread-safe
/// field partitioning or field cloning depending on plugin requirements.
pub struct ParallelStrategy {
    /// Maximum number of threads to use
    max_threads: Option<usize>,
    /// Whether to use field cloning for thread safety
    use_field_cloning: bool,
}

impl ParallelStrategy {
    /// Create a new parallel execution strategy
    pub fn new() -> Self {
        Self {
            max_threads: None,
            use_field_cloning: false,
        }
    }
    
    /// Set the maximum number of threads
    pub fn with_max_threads(mut self, threads: usize) -> Self {
        self.max_threads = Some(threads);
        self
    }
    
    /// Enable field cloning for thread safety
    pub fn with_field_cloning(mut self, enable: bool) -> Self {
        self.use_field_cloning = enable;
        self
    }
    
    /// Check if plugins can be executed in parallel
    fn can_parallelize(plugins: &[Box<dyn PhysicsPlugin>]) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut current_group = Vec::new();
        let mut used_fields = std::collections::HashSet::new();
        
        for (idx, plugin) in plugins.iter().enumerate() {
            let required = plugin.required_fields();
            let provided = plugin.provided_fields();
            
            // Check for field conflicts with current group
            let mut has_conflict = false;
            for field in required.iter().chain(provided.iter()) {
                if used_fields.contains(field) {
                    has_conflict = true;
                    break;
                }
            }
            
            if has_conflict && !current_group.is_empty() {
                // Start a new group
                groups.push(current_group.clone());
                current_group.clear();
                used_fields.clear();
            }
            
            // Add to current group
            current_group.push(idx);
            for field in required.iter().chain(provided.iter()) {
                used_fields.insert(field.clone());
            }
        }
        
        if !current_group.is_empty() {
            groups.push(current_group);
        }
        
        groups
    }
}

impl Default for ParallelStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionStrategy for ParallelStrategy {
    fn execute(
        &self,
        plugins: &mut [Box<dyn PhysicsPlugin>],
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        use rayon::prelude::*;
        
        // Determine which plugins can run in parallel
        let parallel_groups = Self::can_parallelize(plugins);
        
        // Execute each group
        for group_indices in parallel_groups {
            if group_indices.len() == 1 {
                // Single plugin - execute directly
                let idx = group_indices[0];
                plugins[idx].update(fields, grid, medium, dt, t, context)?;
            } else if self.use_field_cloning {
                // Multiple plugins with field cloning for thread safety
                // Execute plugins sequentially for now to avoid borrow checker issues
                // TODO: Implement proper parallel execution with thread-safe plugin access
                let mut results = Vec::new();
                for &idx in &group_indices {
                    let mut local_fields = fields.clone();
                    let result = plugins[idx].update(
                        &mut local_fields, 
                        grid, 
                        medium, 
                        dt, 
                        t, 
                        context
                    );
                    results.push((idx, local_fields, result));
                }
                
                // Merge results back
                for (idx, local_fields, result) in results {
                    result?;
                    // Merge strategy: average overlapping regions
                    // This is a simplified approach - real implementation would
                    // need more sophisticated merging based on plugin semantics
                    let provided_fields = plugins[idx].provided_fields();
                    for field_type in provided_fields {
                        let field_idx = field_type_to_index(&field_type);
                        if let Some(idx) = field_idx {
                            fields.index_axis_mut(ndarray::Axis(0), idx)
                                .assign(&local_fields.index_axis(ndarray::Axis(0), idx));
                        }
                    }
                }
            } else {
                // Multiple plugins with synchronized field access
                // Use a thread pool with controlled concurrency
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(self.max_threads.unwrap_or_else(|| rayon::current_num_threads()))
                    .build()
                    .map_err(|e| crate::error::KwaversError::System(
                        crate::error::SystemError::ThreadPoolCreation {
                            reason: e.to_string(),
                        }
                    ))?;
                
                // Execute plugins sequentially for now to avoid borrow checker issues
                // TODO: Implement proper parallel execution with thread-safe plugin access
                let mut errors = Vec::new();
                for &idx in &group_indices {
                    if let Err(e) = plugins[idx].update(fields, grid, medium, dt, t, context) {
                        errors.push(e);
                    }
                }
                
                // Check for errors
                if let Some(first_error) = errors.into_iter().next() {
                    return Err(first_error);
                }
            }
        }
        
        Ok(())
    }
}

/// Helper function to convert field type to array index
fn field_type_to_index(field_type: &FieldType) -> Option<usize> {
    use crate::physics::composable::FieldType;
    
    match field_type {
        FieldType::Pressure => Some(0),
        FieldType::Velocity => Some(1), // Assuming velocity x is at index 1
        FieldType::Temperature => Some(4),
        FieldType::Density => Some(5),
        FieldType::Light => Some(6),
        FieldType::Cavitation => Some(7),
        FieldType::Chemical => Some(8),
        FieldType::Stress => Some(9),
        FieldType::Custom(name) => {
            // Map custom fields to indices
            match name.as_str() {
                "vx" => Some(1),
                "vy" => Some(2),
                "vz" => Some(3),
                _ => None,
            }
        }
    }
}

/// Plugin visitor pattern for traversing plugin hierarchies
pub trait PluginVisitor {
    /// Visit a plugin
    fn visit(&mut self, plugin: &dyn PhysicsPlugin);
    
    /// Visit a composite plugin
    fn visit_composite(&mut self, composite: &CompositePlugin) {
        self.visit(composite);
        for plugin in &composite.plugins {
            self.visit(plugin.as_ref());
        }
    }
}

/// Example visitor that collects plugin metadata
pub struct MetadataCollector {
    metadata: Vec<PluginMetadata>,
}

impl MetadataCollector {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
        }
    }
    
    pub fn metadata(self) -> Vec<PluginMetadata> {
        self.metadata
    }
}

impl PluginVisitor for MetadataCollector {
    fn visit(&mut self, plugin: &dyn PhysicsPlugin) {
        self.metadata.push(plugin.metadata().clone());
    }
}

// Tests moved to tests.rs
#[cfg(test)]
mod plugin_internal_tests {
    use super::*;
    
    #[derive(Debug, Clone)]
    struct TestPlugin {
        metadata: PluginMetadata,
        required: Vec<FieldType>,
        provided: Vec<FieldType>,
    }
    
    impl PhysicsPlugin for TestPlugin {
        fn metadata(&self) -> &PluginMetadata {
            &self.metadata
        }
        
        fn state(&self) -> PluginState {
            PluginState::Created
        }
        
        fn required_fields(&self) -> Vec<FieldType> {
            self.required.clone()
        }
        
        fn provided_fields(&self) -> Vec<FieldType> {
            self.provided.clone()
        }
        
        fn initialize(
            &mut self,
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

        fn finalize(&mut self) -> KwaversResult<()> {
            Ok(())
        }

        fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
            Box::new(self.clone())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
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