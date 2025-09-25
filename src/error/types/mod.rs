//! Error type hierarchical organization
//!
//! Deep vertical decomposition of error types following domain boundaries:
//! - Domain: Physics, configuration, and validation specific errors
//! - System: I/O, grid, and infrastructure errors

pub mod domain;
pub mod system;

// Re-export for hierarchical access
pub use domain::*;
pub use system::*;