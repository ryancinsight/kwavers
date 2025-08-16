//! Grid-related error types

use std::error::Error as StdError;
use std::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GridError {
    InvalidDimensions {
        nx: usize,
        ny: usize,
        nz: usize,
        reason: String,
    },
    InvalidSpacing {
        dx: f64,
        dy: f64,
        dz: f64,
        reason: String,
    },
    OutOfMemory {
        required_bytes: usize,
        available_bytes: usize,
    },
    IndexOutOfBounds {
        index: (usize, usize, usize),
        dimensions: (usize, usize, usize),
    },
    NotInitialized,
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { nx, ny, nz, reason } => {
                write!(f, "Invalid grid dimensions ({}, {}, {}): {}", nx, ny, nz, reason)
            }
            Self::InvalidSpacing { dx, dy, dz, reason } => {
                write!(f, "Invalid grid spacing ({}, {}, {}): {}", dx, dy, dz, reason)
            }
            Self::OutOfMemory { required_bytes, available_bytes } => {
                write!(f, "Grid requires {} bytes but only {} available", required_bytes, available_bytes)
            }
            Self::IndexOutOfBounds { index, dimensions } => {
                write!(f, "Index {:?} out of bounds for dimensions {:?}", index, dimensions)
            }
            Self::NotInitialized => write!(f, "Grid not initialized"),
        }
    }
}

impl StdError for GridError {}
