//! Configuration error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

/// Configuration-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigError {
    /// Missing required parameter
    MissingParameter { parameter: String, section: String },
    /// Invalid parameter value
    InvalidValue {
        parameter: String,
        value: String,
        constraint: String,
    },
    /// Configuration file not found
    FileNotFound { path: String },
    /// Parse error in configuration
    ParseError { line: usize, message: String },
    /// Validation failed
    ValidationFailed {
        field: String,
        value: String,
        constraint: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingParameter { parameter, section } => {
                write!(f, "Missing parameter '{parameter}' in section '{section}'")
            }
            Self::InvalidValue {
                parameter,
                value,
                constraint,
            } => {
                write!(f, "Invalid value for '{parameter}': {value} ({constraint})")
            }
            Self::FileNotFound { path } => {
                write!(f, "Configuration file not found: {path}")
            }
            Self::ParseError { line, message } => {
                write!(f, "Parse error at line {line}: {message}")
            }
            Self::ValidationFailed {
                field,
                value,
                constraint,
            } => {
                write!(
                    f,
                    "Validation failed for {field}: {value} violates {constraint}"
                )
            }
        }
    }
}

impl StdError for ConfigError {}
