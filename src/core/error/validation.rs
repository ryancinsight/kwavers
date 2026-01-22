//! Validation error types

use thiserror::Error;

/// Validation errors for input parameters and configurations
#[derive(Debug, Clone, Error)]
pub enum ValidationError {
    /// Field validation failed
    #[error("Field '{field}' validation failed: value '{value}' {constraint}")]
    FieldValidation {
        field: String,
        value: String,
        constraint: String,
    },

    /// Range validation failed
    #[error("Value {value} is out of range [{min}, {max}]")]
    OutOfRange { value: f64, min: f64, max: f64 },

    /// Required field is missing
    #[error("Required field '{field}' is missing")]
    MissingField { field: String },

    /// Invalid format
    #[error("Invalid format for '{field}': expected {expected}, got {actual}")]
    InvalidFormat {
        field: String,
        expected: String,
        actual: String,
    },

    /// Constraint violation
    #[error("Constraint violation: {message}")]
    ConstraintViolation { message: String },

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, actual {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Invalid value
    #[error("Invalid value for parameter '{parameter}': {value} - {reason}")]
    InvalidValue {
        parameter: String,
        value: f64,
        reason: String,
    },

    /// Invalid parameter
    #[error("Invalid parameter '{parameter}': {reason}")]
    InvalidParameter { parameter: String, reason: String },
}
