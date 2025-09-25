//! Grid-specific error types
//!
//! Computational grid and memory layout error handling

use thiserror::Error;

/// Grid operation error types
#[derive(Error, Debug, Clone)]
pub enum GridErrorType {
    #[error("Grid dimension error: invalid dimensions {nx}x{ny}x{nz}")]
    InvalidDimensions { nx: usize, ny: usize, nz: usize },
    
    #[error("Grid spacing error: invalid spacing dx={dx}, dy={dy}, dz={dz}")]
    InvalidSpacing { dx: f64, dy: f64, dz: f64 },
    
    #[error("Grid index out of bounds: ({x}, {y}, {z}) not in [0, {max_x}] × [0, {max_y}] × [0, {max_z}]")]
    IndexOutOfBounds {
        x: usize,
        y: usize,
        z: usize,
        max_x: usize,
        max_y: usize,
        max_z: usize,
    },
    
    #[error("Memory allocation failed: requested {size_bytes} bytes for grid")]
    AllocationFailed { size_bytes: usize },
    
    #[error("Grid compatibility error: grids have incompatible dimensions")]
    IncompatibleGrids,
    
    #[error("Grid initialization failed: {reason}")]
    InitializationFailed { reason: String },
}