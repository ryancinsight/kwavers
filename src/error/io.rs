//! Data I/O error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

/// Data I/O errors
#[derive(Debug, Clone, Serialize, Deserialize))]
pub enum DataError {
    /// I/O error
    IoError(String),
    /// Format error
    FormatError { format: String, reason: String },
    /// Dimension mismatch
    DimensionMismatch {
        expected: (usize, usize, usize),
        actual: (usize, usize, usize),
    },
    /// Insufficient data
    InsufficientData { required: usize, available: usize },
    /// Invalid format
    InvalidFormat { format: String, reason: String },
    /// File not found
    FileNotFound { path: String },
    /// Data corruption
    Corruption { location: String, reason: String },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::FormatError { format, reason } => {
                write!(f, "Format error in {}: {}", format, reason)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            Self::InsufficientData {
                required,
                available,
            } => {
                write!(
                    f,
                    "Insufficient data: required {}, available {}",
                    required, available
                )
            }
            Self::InvalidFormat { format, reason } => {
                write!(f, "Invalid format {}: {}", format, reason)
            }
            Self::FileNotFound { path } => {
                write!(f, "File not found: {}", path)
            }
            Self::Corruption { location, reason } => {
                write!(f, "Data corruption at {}: {}", location, reason)
            }
        }
    }
}

impl StdError for DataError {}
