//! Comprehensive error handling system for kwavers
//!
//! This module provides a structured error handling system organized by domain:
//! - `physics`: Physics simulation errors
//! - `gpu`: GPU and acceleration errors
//! - `io`: Data I/O and format errors
//! - `config`: Configuration and validation errors
//! - `numerical`: Numerical computation errors
//! - `field`: Field registry and management errors

pub mod composite;
pub mod config;
pub mod context;
pub mod field;
pub mod grid;
pub mod io;
pub mod medium;
pub mod numerical;
pub mod physics;
pub mod system;
pub mod validation;

// Re-export main error types
pub use composite::CompositeError;
pub use config::ConfigError;
pub use context::ErrorContext;
pub use field::FieldError;
pub use grid::GridError;
pub use io::DataError;
pub use medium::MediumError;
pub use numerical::NumericalError;
pub use physics::PhysicsError;
pub use system::SystemError;
pub use validation::ValidationError;

use thiserror::Error;

/// Main error type for kwavers operations
///
/// This enum uses thiserror for automatic, correct implementations of
/// Display and Error traits, preserving the full error chain.
#[derive(Debug, Error)]
pub enum KwaversError {
    /// Grid-related errors
    #[error(transparent)]
    Grid(#[from] GridError),

    /// Medium-related errors
    #[error(transparent)]
    Medium(#[from] MediumError),

    /// Physics simulation errors
    #[error(transparent)]
    Physics(#[from] PhysicsError),

    /// Data I/O and format errors
    #[error(transparent)]
    Data(#[from] DataError),

    /// Configuration errors
    #[error(transparent)]
    Config(#[from] ConfigError),

    /// Numerical computation errors
    #[error(transparent)]
    Numerical(#[from] NumericalError),

    /// Field registry errors
    #[error(transparent)]
    Field(#[from] FieldError),

    /// System errors
    #[error(transparent)]
    System(#[from] SystemError),

    /// Validation errors
    #[error(transparent)]
    Validation(#[from] ValidationError),

    /// Standard I/O errors
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    /// NIFTI format errors
    #[cfg(feature = "nifti")]
    #[error("NIFTI format error")]
    Nifti(#[from] nifti::error::NiftiError),

    /// Feature not yet implemented
    #[error("Feature not yet implemented: {0}")]
    NotImplemented(String),

    /// GPU computation errors
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Invalid input (for backward compatibility, prefer specific error types)
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Result type alias for operations that may return a KwaversError
pub type KwaversResult<T> = Result<T, KwaversError>;

// Backward compatibility conversions
impl From<String> for KwaversError {
    fn from(s: String) -> Self {
        KwaversError::InvalidInput(s)
    }
}

impl From<&str> for KwaversError {
    fn from(s: &str) -> Self {
        KwaversError::InvalidInput(s.to_string())
    }
}
