//! System error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemError {
    MemoryAllocation {
        requested_bytes: usize,
        reason: String,
    },
    ThreadCreation {
        reason: String,
    },
    ThreadPoolCreation {
        reason: String,
    },
    ResourceExhausted {
        resource: String,
        reason: String,
    },
    Io {
        operation: String,
        reason: String,
    },
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryAllocation {
                requested_bytes,
                reason,
            } => {
                write!(
                    f,
                    "Memory allocation failed for {} bytes: {}",
                    requested_bytes, reason
                )
            }
            Self::ThreadCreation { reason } => {
                write!(f, "Thread creation failed: {}", reason)
            }
            Self::ThreadPoolCreation { reason } => {
                write!(f, "Thread pool creation failed: {}", reason)
            }
            Self::ResourceExhausted { resource, reason } => {
                write!(f, "System resource '{}' exhausted: {}", resource, reason)
            }
            Self::Io { operation, reason } => {
                write!(f, "IO operation '{}' failed: {}", operation, reason)
            }
        }
    }
}

impl StdError for SystemError {}
