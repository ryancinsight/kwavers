// domain/sensor/recorder/traits.rs - Core recorder traits

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array4;

/// Trait for data recording (Dependency Inversion Principle)
pub trait RecorderTrait: Send + Sync {
    /// Initialize the recorder
    fn initialize(&mut self, grid: &Grid) -> KwaversResult<()>;

    /// Record data at a specific time step
    fn record(&mut self, fields: &Array4<f64>, step: usize) -> KwaversResult<()>;

    /// Finalize recording and save data
    fn finalize(&mut self) -> KwaversResult<()>;
}
