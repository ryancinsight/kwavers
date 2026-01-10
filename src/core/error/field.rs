//! Field registry error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldError {
    NotRegistered(String),
    Inactive(String),
    DataNotInitialized,
    DimensionMismatch {
        field: String,
        expected: (usize, usize, usize),
        actual: (usize, usize, usize),
    },
    InvalidFieldAccess {
        field: String,
        reason: String,
    },
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRegistered(name) => write!(f, "Field not registered: {name}"),
            Self::Inactive(name) => write!(f, "Field inactive: {name}"),
            Self::DataNotInitialized => write!(f, "Field data not initialized"),
            Self::DimensionMismatch {
                field,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Field {field} dimension mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::InvalidFieldAccess { field, reason } => {
                write!(f, "Invalid field access for {field}: {reason}")
            }
        }
    }
}

impl StdError for FieldError {}
