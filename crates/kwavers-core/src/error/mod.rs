//! Comprehensive error handling system for kwavers
//!
//! This module provides a structured error handling system organized by domain:
//!
//! ## Not yet implemented
//!
//! - **Recovery strategies**: Automatic fallback and graceful degradation on component failure.
//! - **Error propagation context**: Causal chain accumulation for root-cause diagnosis.
//! - **Diagnostic classification**: Automated error classification and root-cause analysis.
//! - **Telemetry integration**: Error monitoring and alerting hooks (Nygard 2007, Release It!).
//! - **Fault injection**: Error simulation for resilience and chaos testing.
//! - **Internationalization**: Locale-aware error message formatting.
//!
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
pub mod gpu;
pub mod io;
pub mod numerical;
pub mod physics;
pub mod system;
pub mod validation;

// Re-export main error types
pub use composite::{CompositeError, MultiError};
pub use config::ConfigError;
pub use context::ErrorContext;
pub use field::FieldError;
pub use io::DataError;
pub use numerical::NumericalError;
pub use physics::PhysicsError;
pub use system::SystemError;
pub use validation::ValidationError;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Specific errors for grid creation and validation
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum GridError {
    /// Grid dimensions must be positive
    #[error("Grid dimensions must be positive, got nx={nx}, ny={ny}, nz={nz}")]
    ZeroDimension { nx: usize, ny: usize, nz: usize },

    /// Grid spacing must be positive
    #[error("Grid spacing must be positive, got dx={dx}, dy={dy}, dz={dz}")]
    NonPositiveSpacing { dx: f64, dy: f64, dz: f64 },

    /// Grid is too large for available memory
    #[error("Grid too large: {nx}x{ny}x{nz} = {total} points exceeds maximum {max}")]
    TooLarge {
        nx: usize,
        ny: usize,
        nz: usize,
        total: usize,
        max: usize,
    },

    /// Grid is too small for the numerical scheme
    #[error(
        "Grid too small: minimum {min} points required in each dimension, got ({nx}, {ny}, {nz})"
    )]
    TooSmall {
        nx: usize,
        ny: usize,
        nz: usize,
        min: usize,
    },

    /// Failed to convert grid spacing to target numeric type
    #[error("Failed to convert grid spacing {value} to {target_type}")]
    GridConversion {
        value: f64,
        target_type: &'static str,
    },

    /// Dimension mismatch between arrays and grid
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum MediumError {
    #[error("Invalid medium property {property} = {value}: {constraint}")]
    InvalidProperties {
        property: String,
        value: f64,
        constraint: String,
    },

    #[error("Medium '{medium_name}' not found")]
    NotFound { medium_name: String },

    #[error("Medium initialization failed: {reason}")]
    InitializationFailed { reason: String },
}

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

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Feature not available or implemented
    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    /// Performance validation error
    #[error("Performance error: {0}")]
    PerformanceError(String),

    /// Standard I/O errors
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    /// NIFTI format errors
    #[cfg(feature = "nifti")]
    #[error("NIFTI format error")]
    Nifti(#[from] nifti::error::NiftiError),

    /// DICOM format errors (always present; DICOM I/O lives in ritk-io,
    /// which is now a mandatory dep — no feature gate required).
    #[error("DICOM format error: {0}")]
    DicomError(String),

    /// Feature not yet implemented
    #[error("Feature not yet implemented: {0}")]
    NotImplemented(String),

    /// GPU computation errors
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Resource limit exceeded (GPU memory, etc.)
    #[error("Resource limit exceeded: {message}")]
    ResourceLimitExceeded {
        /// Error message
        message: String,
    },

    /// Invalid input (for backward compatibility, prefer specific error types)
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Multiple errors occurred during validation
    #[error(transparent)]
    MultipleErrors(#[from] MultiError),

    /// Concurrency error (lock acquisition failure)
    #[error("Concurrency error: {message}")]
    ConcurrencyError {
        /// Error message
        message: String,
    },

    /// Visualization error
    #[error("Visualization error: {message}")]
    Visualization {
        /// Error message
        message: String,
    },

    /// Array shape errors
    #[error("Shape error: {0}")]
    Shape(String),

    /// Anyhow errors
    #[error(transparent)]
    Other(#[from] anyhow::Error),

    /// Leto (array/layout) errors
    #[error("Leto error: {0}")]
    Leto(String),
}

impl From<leto::LetoError> for KwaversError {
    fn from(err: leto::LetoError) -> Self {
        Self::Leto(err.to_string())
    }
}

/// Result type alias for operations that may return a `KwaversError`
pub type KwaversResult<T> = Result<T, KwaversError>;

// Backward compatibility conversions
impl From<String> for KwaversError {
    fn from(s: String) -> Self {
        Self::InvalidInput(s)
    }
}

impl From<&str> for KwaversError {
    fn from(s: &str) -> Self {
        Self::InvalidInput(s.to_owned())
    }
}

#[cfg(feature = "channels")]
impl From<flume::RecvError> for KwaversError {
    fn from(err: flume::RecvError) -> Self {
        Self::GpuError(format!("Channel receive error: {}", err))
    }
}

#[cfg(feature = "registration")]
impl From<ritk_registration::RegistrationError> for KwaversError {
    fn from(err: ritk_registration::RegistrationError) -> Self {
        Self::InvalidInput(err.to_string())
    }
}

#[cfg(feature = "registration")]
impl From<ritk_registration::classical::RegistrationError> for KwaversError {
    fn from(err: ritk_registration::classical::RegistrationError) -> Self {
        Self::InvalidInput(err.to_string())
    }
}
