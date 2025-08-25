//! Plugin-based solver architecture following SOLID principles
//!
//! This module provides a clean, extensible solver that:
//! - Follows Single Responsibility Principle (orchestration only)
//! - Applies Dependency Inversion (depends on abstractions)
//! - Is Open/Closed (extensible via plugins)

mod field_registry;
mod field_provider;
mod performance;
mod solver;

pub use field_registry::FieldRegistry;
pub use field_provider::FieldProvider;
pub use performance::PerformanceMonitor;
pub use solver::PluginBasedSolver;

// Re-export for backward compatibility
pub use solver::PluginBasedSolver as Solver;