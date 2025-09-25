//! Domain-specific error types
//!
//! Physics, configuration, and validation error hierarchies

pub mod physics;
pub mod config;
pub mod validation;

// Re-export domain errors
pub use physics::*;
pub use config::*;
pub use validation::*;