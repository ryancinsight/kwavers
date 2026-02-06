//! Domain-specific error types
//!
//! Physics, configuration, and validation error hierarchies

pub mod config;
pub mod physics;
pub mod validation;

// Explicit re-exports of domain error types
pub use config::ConfigErrorType;
pub use physics::PhysicsErrorType;
pub use validation::ValidationErrorType;
