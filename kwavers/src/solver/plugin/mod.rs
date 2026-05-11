//! Plugin orchestration for solvers
//!
//! This module provides the execution engine and management logic for running plugins
//! within the solver loop.

pub mod execution;
pub mod manager;

pub use execution::{ExecutionStrategy, ParallelStrategy, SequentialStrategy};
pub use manager::PluginManager;
