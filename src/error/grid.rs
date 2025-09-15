//! Grid-specific error types

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Specific errors for grid creation and validation
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum GridError {
    /// Grid dimensions must be positive
    #[error("Grid dimensions must be positive, got nx={nx}, ny={ny}, nz={nz}")]
    ZeroDimension { nx: usize, ny: usize, nz: usize },

    /// Grid spacing must be positive
    #[error("Grid spacing must be positive, got dx={dx}, dy={dy}, dz={dz}")]
    NonPositiveSpacing { dx: f64, dy: f64, dz: f64 },

    /// Grid is too large for available memory
    #[error("Grid too large: {nx}x{ny}x{nz} = {total} points exceeds maximum {max}")]
    TooLarge {
        nx: usize,
        ny: usize,
        nz: usize,
        total: usize,
        max: usize,
    },

    /// Grid is too small for the numerical scheme
    #[error(
        "Grid too small: minimum {min} points required in each dimension, got ({nx}, {ny}, {nz})"
    )]
    TooSmall {
        nx: usize,
        ny: usize,
        nz: usize,
        min: usize,
    },

    /// Failed to convert grid spacing to target numeric type
    #[error("Failed to convert grid spacing {value} to {target_type}")]
    GridConversion {
        value: f64,
        target_type: &'static str,
    },
}

// Note: From<GridError> for KwaversError is not needed as GridError is already
// part of the KwaversError enum via the Grid variant
