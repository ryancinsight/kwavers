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
pub mod gpu;
pub mod grid;
pub mod io;
pub mod medium;
pub mod numerical;
pub mod physics;
pub mod system;

// Re-export main error types
pub use composite::CompositeError;
pub use config::ConfigError;
pub use context::ErrorContext;
pub use field::FieldError;
pub use gpu::GpuError;
pub use grid::GridError;
pub use io::DataError;
pub use medium::MediumError;
pub use numerical::NumericalError;
pub use physics::PhysicsError;
pub use system::SystemError;

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

/// Main error type for kwavers operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KwaversError {
    /// Grid-related errors
    Grid(GridError),
    /// Medium-related errors
    Medium(MediumError),
    /// Physics simulation errors
    Physics(PhysicsError),
    /// GPU acceleration errors
    Gpu(GpuError),
    /// Data I/O and format errors
    Data(DataError),
    /// Configuration errors
    Config(ConfigError),
    /// Numerical computation errors
    Numerical(NumericalError),
    /// Field registry errors
    Field(FieldError),
    /// System errors
    System(SystemError),
    /// Composite error with multiple underlying errors
    Composite(CompositeError),
    /// Validation errors
    Validation(ValidationError),
    /// Invalid input parameters
    InvalidInput(String),
    /// Invalid parameter error
    InvalidParameter(String),
    /// Numerical error (computation issues)
    NumericalError(String),
    /// Invalid state error
    InvalidState(String),
    /// IO errors
    Io(String),
    /// Concurrency errors
    ConcurrencyError {
        operation: String,
        resource: String,
        reason: String,
    },
    /// Feature not yet implemented
    NotImplemented(String),
}

impl fmt::Display for KwaversError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grid(e) => write!(f, "Grid error: {}", e),
            Self::Medium(e) => write!(f, "Medium error: {}", e),
            Self::Physics(e) => write!(f, "Physics error: {}", e),
            Self::Gpu(e) => write!(f, "GPU error: {}", e),
            Self::Data(e) => write!(f, "Data error: {}", e),
            Self::Config(e) => write!(f, "Configuration error: {}", e),
            Self::Numerical(e) => write!(f, "Numerical error: {}", e),
            Self::Field(e) => write!(f, "Field error: {}", e),
            Self::System(e) => write!(f, "System error: {}", e),
            Self::Composite(e) => write!(f, "Multiple errors: {}", e),
            Self::Validation(e) => write!(f, "Validation error: {}", e),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Self::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            Self::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
            Self::Io(msg) => write!(f, "IO error: {}", msg),
            Self::ConcurrencyError {
                operation,
                resource,
                reason,
            } => write!(
                f,
                "Concurrency error during '{}' on '{}': {}",
                operation, resource, reason
            ),
            Self::NotImplemented(feature) => write!(f, "Not implemented: {}", feature),
        }
    }
}

impl StdError for KwaversError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Grid(e) => Some(e),
            Self::Medium(e) => Some(e),
            Self::Physics(e) => Some(e),
            Self::Gpu(e) => Some(e),
            Self::Data(e) => Some(e),
            Self::Config(e) => Some(e),
            Self::Numerical(e) => Some(e),
            Self::Field(e) => Some(e),
            Self::System(e) => Some(e),
            Self::Composite(e) => Some(e),
            Self::Validation(e) => Some(e),
            Self::InvalidInput(_) => None,
            Self::InvalidParameter(_) => None,
            Self::NumericalError(_) => None,
            Self::InvalidState(_) => None,
            Self::Io(_) => None,
            Self::ConcurrencyError { .. } => None,
            Self::NotImplemented(_) => None,
        }
    }
}

/// Result type alias for kwavers operations
pub type KwaversResult<T> = Result<T, KwaversError>;

// Conversion implementations
impl From<PhysicsError> for KwaversError {
    fn from(error: PhysicsError) -> Self {
        Self::Physics(error)
    }
}

impl From<GpuError> for KwaversError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

impl From<DataError> for KwaversError {
    fn from(error: DataError) -> Self {
        Self::Data(error)
    }
}

impl From<ConfigError> for KwaversError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<NumericalError> for KwaversError {
    fn from(error: NumericalError) -> Self {
        Self::Numerical(error)
    }
}

impl From<FieldError> for KwaversError {
    fn from(error: FieldError) -> Self {
        Self::Field(error)
    }
}

impl From<std::io::Error> for KwaversError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl From<GridError> for KwaversError {
    fn from(error: GridError) -> Self {
        Self::Grid(error)
    }
}

impl From<MediumError> for KwaversError {
    fn from(error: MediumError) -> Self {
        Self::Medium(error)
    }
}

impl From<SystemError> for KwaversError {
    fn from(error: SystemError) -> Self {
        Self::System(error)
    }
}

/// Validation error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    /// Field validation failed
    FieldValidation {
        field: String,
        value: String,
        constraint: String,
    },
    /// Range validation failed
    RangeValidation {
        field: String,
        value: String,
        min: String,
        max: String,
    },
    /// State validation failed
    StateValidation,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldValidation {
                field,
                value,
                constraint,
            } => {
                write!(
                    f,
                    "Field validation failed for {}: {} violates {}",
                    field, value, constraint
                )
            }
            Self::RangeValidation {
                field,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "Range validation failed for {}: {} not in [{}, {}]",
                    field, value, min, max
                )
            }
            Self::StateValidation => {
                write!(f, "State validation failed")
            }
        }
    }
}

impl StdError for ValidationError {}

impl From<ValidationError> for KwaversError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}
