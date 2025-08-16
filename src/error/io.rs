//! Data I/O error types

use std::error::Error as StdError;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Data I/O errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataError {
    /// I/O error
    IoError(String),
    /// Format error
    FormatError {
        format: String,
        reason: String,
    },
    /// Dimension mismatch
    DimensionMismatch {
        expected: (usize, usize, usize),
        actual: (usize, usize, usize),
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::FormatError { format, reason } => {
                write!(f, "Format error in {}: {}", format, reason)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {:?}, got {:?}", expected, actual)
            }
        }
    }
}

impl StdError for DataError {}