//! Plugin execution strategies
//!
//! This module provides different execution strategies for running plugins.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use crate::plugin::{Plugin, PluginContext};
use ndarray::Array4;

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

/// Parallel execution strategy
///
/// Note: This currently executes sequentially due to mutable field access constraints.
/// True parallelism would require architectural changes (read/write phase separation).
#[derive(Debug)]
pub struct ParallelStrategy {
    // Future: thread_pool field will be added when parallel execution is implemented
}

impl ParallelStrategy {
    /// Create a new parallel execution strategy
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Create with a specific thread pool (future implementation)
    #[must_use]
    pub fn with_thread_pool(_pool: rayon::ThreadPool) -> Self {
        Self {}
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
        // Note: Currently executes sequentially due to mutable field access
        // True parallelism would require:
        // 1. Read phase: plugins declare what they need to read
        // 2. Compute phase: parallel computation
        // 3. Write phase: merge results

        for plugin in plugins.iter_mut() {
            plugin.update(fields, grid, medium, dt, t, context)?;
        }
        Ok(())
    }
}
