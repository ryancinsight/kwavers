use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;
use std::fmt::Debug;

/// Trait for acoustic streaming models.
///
/// Implementors of this trait simulate the fluid motion induced by acoustic waves.
pub trait StreamingModelTrait: Debug + Send + Sync {
    /// Advances the streaming velocity field by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current acoustic pressure field.
    /// * `grid` - Simulation grid.
    /// * `medium` - Medium properties.
    /// * `dt` - Time step.
    fn update_velocity(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    );

    /// Returns a reference to the 3D array of the streaming velocity field (m/s).
    fn velocity(&self) -> &Array3<f64>;
}
