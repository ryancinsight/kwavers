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
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRegistered(name) => write!(f, "Field not registered: {}", name),
            Self::Inactive(name) => write!(f, "Field inactive: {}", name),
            Self::DataNotInitialized => write!(f, "Field data not initialized"),
            Self::DimensionMismatch {
                field,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Field {} dimension mismatch: expected {:?}, got {:?}",
                    field, expected, actual
                )
            }
        }
    }
}

impl StdError for FieldError {}
