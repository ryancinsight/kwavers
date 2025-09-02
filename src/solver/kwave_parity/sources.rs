//! Source term implementations for k-Wave compatibility
//!
//! Supports various source types with smoothing

use ndarray::Array3;

/// Source types supported by k-Wave
#[derive(Debug, Clone)]
pub enum SourceType {
    /// Pressure source
    Pressure,
    /// Velocity source
    Velocity,
    /// Mass source
    MassSource,
    /// Transducer source
    Transducer,
}
