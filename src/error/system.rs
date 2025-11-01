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
    ResourceUnavailable {
        resource: String,
    },
    InvalidOperation {
        operation: String,
        reason: String,
    },
    Io {
        operation: String,
        reason: String,
    },
    InvalidConfiguration {
        parameter: String,
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
                    "Memory allocation failed for {requested_bytes} bytes: {reason}"
                )
            }
            Self::ThreadCreation { reason } => {
                write!(f, "Thread creation failed: {reason}")
            }
            Self::ThreadPoolCreation { reason } => {
                write!(f, "Thread pool creation failed: {reason}")
            }
            Self::ResourceExhausted { resource, reason } => {
                write!(f, "System resource '{resource}' exhausted: {reason}")
            }
            Self::ResourceUnavailable { resource } => {
                write!(f, "System resource '{resource}' unavailable")
            }
            Self::InvalidOperation { operation, reason } => {
                write!(f, "Invalid operation '{operation}': {reason}")
            }
            Self::Io { operation, reason } => {
                write!(f, "IO operation '{operation}' failed: {reason}")
            }
            Self::InvalidConfiguration { parameter, reason } => {
                write!(f, "Invalid configuration for parameter '{parameter}': {reason}")
            }
        }
    }
}

impl StdError for SystemError {}
