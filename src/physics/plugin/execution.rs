//! Plugin execution strategies
//!
//! This module provides different execution strategies for running plugins.

use super::{PhysicsPlugin, PluginContext};
use crate::error::{KwaversResult, KwaversError, ValidationError};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array4;
use std::sync::Arc;
use rayon::prelude::*;

/// Strategy for executing plugins
pub trait ExecutionStrategy: Send + Sync {
    /// Execute a collection of plugins
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
#[deprecated(note = "Currently executes sequentially due to &mut fields constraint")]
pub struct ParallelStrategy {
    thread_pool: Option<rayon::ThreadPool>,
}

impl ParallelStrategy {
    /// Create a new parallel execution strategy
    pub fn new() -> Self {
        Self {
            thread_pool: None,
        }
    }
    
    /// Create with a specific thread pool
    pub fn with_thread_pool(pool: rayon::ThreadPool) -> Self {
        Self {
            thread_pool: Some(pool),
        }
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