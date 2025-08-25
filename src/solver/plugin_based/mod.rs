//! Plugin-based solver architecture following SOLID principles
//!
//! This module provides a clean, extensible solver that:
//! - Follows Single Responsibility Principle (orchestration only)
//! - Applies Dependency Inversion (depends on abstractions)
//! - Is Open/Closed (extensible via plugins)
//! - Uses dynamic field registration (no static const indices)

pub mod field_registry;
pub mod field_provider;
pub mod solver;
pub mod performance;

// Re-export main types
pub use field_registry::FieldRegistry;
pub use field_provider::FieldProvider;
pub use solver::PluginBasedSolver;
pub use performance::PerformanceMonitor;