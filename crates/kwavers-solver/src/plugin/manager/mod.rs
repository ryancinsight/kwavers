//! Plugin manager for orchestrating plugin lifecycle and execution.

mod dependency;
mod execution_impl;

use super::execution::{ExecutionStrategy, SequentialStrategy};
use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_domain::plugin::{Plugin, PluginFields};
use ndarray::Array3;
use std::collections::HashMap;

/// Per-plugin and total execution timings.
#[derive(Debug, Clone, Default)]
pub struct PluginExecutionMetrics {
    plugin_times: HashMap<String, std::time::Duration>,
    total_time: std::time::Duration,
}

impl PluginExecutionMetrics {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_plugin_execution(&mut self, plugin_id: &str, duration: std::time::Duration) {
        self.plugin_times.insert(plugin_id.to_owned(), duration);
    }

    pub fn record_total_execution(&mut self, duration: std::time::Duration) {
        self.total_time = duration;
    }

    #[must_use]
    pub fn get_plugin_time(&self, plugin_id: &str) -> Option<std::time::Duration> {
        self.plugin_times.get(plugin_id).copied()
    }

    #[must_use]
    pub fn get_total_time(&self) -> std::time::Duration {
        self.total_time
    }

    #[must_use]
    pub fn get_all_plugin_times(&self) -> &HashMap<String, std::time::Duration> {
        &self.plugin_times
    }
}

/// Plugin manager: coordinates plugin lifecycle, dependency ordering, and execution.
pub struct PluginManager {
    pub(self) plugins: Vec<Box<dyn Plugin>>,
    pub(self) execution_order: Vec<usize>,
    pub(self) execution_strategy: Box<dyn ExecutionStrategy>,
    pub(self) extra_fields: PluginFields,
    pub(self) performance_metrics: PluginExecutionMetrics,
}

impl std::fmt::Debug for PluginManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginManager")
            .field("plugins_count", &self.plugins.len())
            .field("execution_order", &self.execution_order)
            .field("execution_strategy", &"<dyn ExecutionStrategy>")
            .field("extra_fields", &self.extra_fields)
            .field("performance_metrics", &self.performance_metrics)
            .finish()
    }
}

impl PluginManager {
    /// Create a new plugin manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            execution_order: Vec::new(),
            execution_strategy: Box::new(SequentialStrategy),
            extra_fields: PluginFields::new(Array3::zeros((1, 1, 1))),
            performance_metrics: PluginExecutionMetrics::new(),
        }
    }

    /// Set the execution strategy.
    pub fn set_execution_strategy(&mut self, strategy: Box<dyn ExecutionStrategy>) {
        self.execution_strategy = strategy;
    }

    /// Initialize all plugins.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn initialize_all(
        &mut self,
        grid: &kwavers_domain::grid::Grid,
        medium: &dyn kwavers_domain::medium::Medium,
    ) -> kwavers_core::error::KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.initialize(grid, medium)?;
        }
        Ok(())
    }

    /// Initialize all plugins (alias for `initialize_all`).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.initialize(grid, medium)?;
        }
        Ok(())
    }

    /// Finalize all plugins.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn finalize(&mut self) -> KwaversResult<()> {
        for plugin in &mut self.plugins {
            plugin.finalize()?;
        }
        Ok(())
    }

    /// Get plugin by index.
    #[must_use]
    pub fn get_plugin(&self, index: usize) -> Option<&dyn Plugin> {
        self.plugins.get(index).map(std::convert::AsRef::as_ref)
    }

    /// Get mutable plugin by index.
    pub fn get_plugin_mut(&mut self, index: usize) -> Option<&mut dyn Plugin> {
        match self.plugins.get_mut(index) {
            Some(plugin) => Some(plugin.as_mut()),
            None => None,
        }
    }

    /// Number of registered plugins.
    #[must_use]
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Alias for `plugin_count`.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.plugins.len()
    }

    /// Dependency-resolved execution order (indices into `plugins`).
    #[must_use]
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    /// Accumulated performance metrics.
    #[must_use]
    pub fn performance_metrics(&self) -> &PluginExecutionMetrics {
        &self.performance_metrics
    }

    /// Iterator over all registered plugins.
    pub fn plugins(&self) -> impl Iterator<Item = &Box<dyn Plugin>> {
        self.plugins.iter()
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}
