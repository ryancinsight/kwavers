//! Processing operations for visualization data

use serde::{Deserialize, Serialize};

/// Data processing operations for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingOperation {
    /// No processing, direct transfer
    None,
    /// Normalize values to [0, 1] range
    Normalize,
    /// Apply logarithmic scaling
    LogScale,
    /// Apply gradient magnitude enhancement
    GradientMagnitude,
    /// Apply 3D Gaussian smoothing
    GaussianSmooth,
    /// Extract isosurface data
    IsosurfaceExtraction,
}

impl ProcessingOperation {
    /// Check if operation requires preprocessing
    pub fn requires_preprocessing(&self) -> bool {
        !matches!(self, ProcessingOperation::None)
    }

    /// Get the computational cost estimate (1-10 scale)
    pub fn cost_estimate(&self) -> u8 {
        match self {
            ProcessingOperation::None => 0,
            ProcessingOperation::Normalize => 1,
            ProcessingOperation::LogScale => 2,
            ProcessingOperation::GradientMagnitude => 5,
            ProcessingOperation::GaussianSmooth => 7,
            ProcessingOperation::IsosurfaceExtraction => 9,
        }
    }
}
