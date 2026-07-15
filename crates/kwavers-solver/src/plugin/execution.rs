//! Plugin execution strategies
//!
//! This module provides different execution strategies for running plugins.

use crate::plugin::{Plugin, PluginContext};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::Array4;

/// Strategy for executing plugins
pub trait ExecutionStrategy: Send + Sync {
    /// Execute a collection of plugins
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        plugins: &mut [Box<dyn Plugin>],
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()>;
}

/// Sequential execution strategy
#[derive(Debug)]
pub struct SequentialStrategy;

impl ExecutionStrategy for SequentialStrategy {
    fn execute(
        &self,
        plugins: &mut [Box<dyn Plugin>],
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        for plugin in plugins.iter_mut() {
            plugin.update(fields, grid, medium, dt, t, context)?;
        }
        Ok(())
    }
}

/// Ordered plugin execution strategy.
///
/// This intentionally executes plugins in order because each plugin receives
/// mutable access to the shared field state. A real parallel implementation
/// needs a read/compute/write phase split before it can preserve plugin
/// semantics without aliasing the field buffers.
#[derive(Debug)]
pub struct ParallelStrategy {
    _private: (),
}

impl ParallelStrategy {
    /// Create a new ordered plugin execution strategy.
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
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
        plugins: &mut [Box<dyn Plugin>],
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        for plugin in plugins.iter_mut() {
            plugin.update(fields, grid, medium, dt, t, context)?;
        }
        Ok(())
    }
}
