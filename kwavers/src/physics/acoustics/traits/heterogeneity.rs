use crate::domain::grid::Grid;
use ndarray::Array3;
use std::fmt::Debug;

/// Trait for models dealing with medium heterogeneity.
///
/// Implementors of this trait provide ways to represent and apply spatial variations
/// in medium properties, such as sound speed.
pub trait HeterogeneityModelTrait: Debug + Send + Sync {
    /// Calculates or returns the adjusted sound speed field considering heterogeneity.
    ///
    /// # Arguments
    ///
    /// * `grid` - Simulation grid.
    ///
    /// # Returns
    ///
    /// A 3D array representing the sound speed at each grid point.
    fn adjust_sound_speed(&self, grid: &Grid) -> Array3<f64>;
}
