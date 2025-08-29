//! Plugin manager for orchestrating plugin lifecycle
//!
//! This module provides the main plugin manager that coordinates plugin execution.

use super::{ExecutionStrategy, Plugin, PluginContext, SequentialStrategy};
use crate::error::{KwaversError, KwaversResult, PhysicsError, ValidationError};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::performance::metrics::PerformanceMetrics;
use crate::physics::field_mapping::UnifiedFieldType;
use ndarray::Array4;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Plugin manager for orchestrating plugin lifecycle and execution
pub struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
    execution_order: Vec<usize>,
    execution_strategy: Box<dyn ExecutionStrategy>,
    context: PluginContext,
    performance_metrics: PerformanceMetrics,
}

impl PluginManager {
    /// Initialize all plugins
    pub fn initialize_all(
        &mut self,
        grid: &crate::grid::Grid,
        medium: &dyn crate::medium::Medium,
    ) -> crate::error::KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.initialize(grid, medium)?;
        }
        Ok(())
    }
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            execution_order: Vec::new(),
            execution_strategy: Box::new(SequentialStrategy),
            context: PluginContext::new(),
            performance_metrics: PerformanceMetrics::new(),
        }
    }

    /// Set the execution strategy
    pub fn set_execution_strategy(&mut self, strategy: Box<dyn ExecutionStrategy>) {
        self.execution_strategy = strategy;
    }

    /// Add a plugin to the manager
    pub fn add_plugin(&mut self, plugin: Box<dyn Plugin>) -> KwaversResult<()> {
        // Check for duplicate plugin IDs
        let new_id = plugin.metadata().id.clone();
        for existing in &self.plugins {
            if existing.metadata().id == new_id {
                return Err(ValidationError::FieldValidation {
                    field: "plugin_id".to_string(),
                    value: new_id.clone(),
                    constraint: "Plugin ID must be unique".to_string(),
                }
                .into());
            }
        }

        self.plugins.push(plugin);
        self.resolve_dependencies()?;
        Ok(())
    }

    /// Initialize all plugins
    pub fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.initialize(grid, medium)?;
        }
        Ok(())
    }

    /// Execute all plugins for one time step
    pub fn execute(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        // Execute plugins in dependency order - safe implementation
        for &idx in &self.execution_order {
            if idx >= self.plugins.len() {
                return Err(KwaversError::Physics(PhysicsError::InvalidState {
                    field: "plugin_index".to_string(),
                    value: idx.to_string(),
                    reason: format!(
                        "Index {} out of bounds for {} plugins",
                        idx,
                        self.plugins.len()
                    ),
                }));
            }

            // Execute plugin directly without unsafe pointer manipulation
            self.plugins[idx].update(fields, grid, medium, dt, t, &self.context)?;
        }

        Ok(())
    }

    /// Execute plugins with performance metrics collection
    pub fn execute_with_metrics(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        // Execute each plugin and measure its time
        for &idx in &self.execution_order {
            if idx >= self.plugins.len() {
                return Err(KwaversError::Physics(PhysicsError::InvalidState {
                    field: "plugin_index".to_string(),
                    value: idx.to_string(),
                    reason: format!(
                        "Index {} out of bounds for {} plugins",
                        idx,
                        self.plugins.len()
                    ),
                }));
            }

            let plugin_start = Instant::now();
            let plugin = &mut self.plugins[idx];

            plugin.update(fields, grid, medium, dt, t, &self.context)?;

            let plugin_duration = plugin_start.elapsed();
            self.performance_metrics
                .record_plugin_execution(&plugin.metadata().id, plugin_duration);
        }

        let total_duration = start.elapsed();
        self.performance_metrics
            .record_total_execution(total_duration);

        Ok(())
    }

    /// Finalize all plugins
    pub fn finalize(&mut self) -> KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.finalize()?;
        }
        Ok(())
    }

    /// Get plugin by index
    pub fn get_plugin(&self, index: usize) -> Option<&dyn Plugin> {
        self.plugins.get(index).map(|p| p.as_ref())
    }

    /// Get mutable plugin by index
    pub fn get_plugin_mut(&mut self, index: usize) -> Option<&mut dyn Plugin> {
        match self.plugins.get_mut(index) {
            Some(plugin) => Some(plugin.as_mut()),
            None => None,
        }
    }

    /// Get number of plugins
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Get the execution order
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Get the number of registered plugins
    pub fn component_count(&self) -> usize {
        self.plugins.len()
    }

    /// Get iterator over plugins
    pub fn plugins(&self) -> impl Iterator<Item = &Box<dyn Plugin>> {
        self.plugins.iter()
    }

    /// Resolve plugin dependencies and determine execution order
    fn resolve_dependencies(&mut self) -> KwaversResult<()> {
        let n = self.plugins.len();
        if n == 0 {
            self.execution_order.clear();
            return Ok(());
        }

        // Build dependency graph
        let mut provides: HashMap<UnifiedFieldType, usize> = HashMap::new();
        let mut requires: Vec<HashSet<UnifiedFieldType>> = Vec::with_capacity(n);

        for (i, plugin) in self.plugins.iter().enumerate() {
            // Record what this plugin provides
            for field in plugin.provided_fields() {
                if let Some(&other) = provides.get(&field) {
                    return Err(ValidationError::FieldValidation {
                        field: "plugin_dependencies".to_string(),
                        value: format!("{:?}", field),
                        constraint: format!(
                            "Field {:?} provided by multiple plugins: {} and {}",
                            field,
                            self.plugins[other].metadata().id,
                            plugin.metadata().id
                        ),
                    }
                    .into());
                }
                provides.insert(field, i);
            }

            // Record what this plugin requires
            let mut deps = HashSet::new();
            for field in plugin.required_fields() {
                deps.insert(field);
            }
            requires.push(deps);
        }

        // Topological sort with cycle detection
        let mut order = Vec::new();
        let mut state = vec![0u8; n]; // 0=unvisited, 1=visiting, 2=visited

        fn visit(
            node: usize,
            state: &mut [u8],
            requires: &[HashSet<UnifiedFieldType>],
            provides: &HashMap<UnifiedFieldType, usize>,
            order: &mut Vec<usize>,
        ) -> Result<(), String> {
            if state[node] == 2 {
                return Ok(()); // Already visited
            }
            if state[node] == 1 {
                return Err(format!("Circular dependency detected at plugin {}", node));
            }

            state[node] = 1; // Mark as visiting

            // Visit dependencies
            for field in &requires[node] {
                if let Some(&dep) = provides.get(field) {
                    if dep != node {
                        visit(dep, state, requires, provides, order)?;
                    }
                }
            }

            state[node] = 2; // Mark as visited
            order.push(node);
            Ok(())
        }

        // Visit all nodes
        for i in 0..n {
            if state[i] == 0 {
                visit(i, &mut state, &requires, &provides, &mut order).map_err(|msg| {
                    ValidationError::FieldValidation {
                        field: "plugin_dependencies".to_string(),
                        value: "invalid".to_string(),
                        constraint: msg,
                    }
                })?;
            }
        }

        self.execution_order = order;
        Ok(())
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}
