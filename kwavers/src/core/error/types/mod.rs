//! Error type hierarchical organization
//!
//! Deep vertical decomposition of error types following domain boundaries:
//! - Domain: Physics, configuration, and validation specific errors
//! - System: I/O, grid, and infrastructure errors

pub mod domain;
pub mod system;

// ============================================================================
// EXPLICIT RE-EXPORTS (Error Type API)
// ============================================================================

// Domain-specific error types
pub use domain::{
    ConfigErrorType,     // Configuration errors
    PhysicsErrorType,    // Physics simulation errors
    ValidationErrorType, // Validation errors
};

// System-level error types
pub use system::{
    GridErrorType,      // Grid-related errors
    IoErrorType,        // I/O operation errors
    NumericalErrorType, // Numerical computation errors
};
