//! Plugin-based solver architecture following SOLID principles
//!
//! This module provides a clean, extensible solver that:
//! - Follows Single Responsibility Principle (orchestration only)
//! - Applies Dependency Inversion (depends on abstractions)
//! - Is Open/Closed (extensible via plugins)

mod field_provider;
mod field_registry;
mod performance;
mod solver;

pub use field_provider::FieldProvider;
pub use field_registry::FieldRegistry;
pub use performance::PerformanceMonitor;
pub use solver::PluginBasedSolver;

// Convenience alias
pub use solver::PluginBasedSolver as Solver;
