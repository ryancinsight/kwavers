// domain/sensor/recorder/traits.rs - Core recorder traits

use kwavers_core::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array4;

/// Trait for data recording (Dependency Inversion Principle)
pub trait RecorderTrait: Send + Sync {
    /// Initialize the recorder
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn initialize(&mut self, grid: &Grid) -> KwaversResult<()>;

    /// Record data at a specific time step
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn record(&mut self, fields: &Array4<f64>, step: usize) -> KwaversResult<()>;

    /// Finalize recording and save data
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn finalize(&mut self) -> KwaversResult<()>;
}
