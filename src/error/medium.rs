//! Medium-related error types

use std::error::Error as StdError;
use std::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediumError {
    InvalidProperties {
        property: String,
        value: f64,
        constraint: String,
    },
    NotFound {
        medium_name: String,
    },
    InitializationFailed {
        reason: String,
    },
}

impl fmt::Display for MediumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProperties { property, value, constraint } => {
                write!(f, "Invalid medium property {} = {}: {}", property, value, constraint)
            }
            Self::NotFound { medium_name } => {
                write!(f, "Medium '{}' not found", medium_name)
            }
            Self::InitializationFailed { reason } => {
                write!(f, "Medium initialization failed: {}", reason)
            }
        }
    }
}

impl StdError for MediumError {}
