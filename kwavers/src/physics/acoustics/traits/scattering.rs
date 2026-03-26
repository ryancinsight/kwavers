use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;
use std::fmt::Debug;

/// Trait for acoustic scattering models.
///
/// Implementors of this trait simulate the scattering of acoustic waves from
/// particles, bubbles, or other inhomogeneities in the medium.
pub trait AcousticScatteringModelTrait: Debug + Send + Sync {
    /// Computes the acoustic scattering effects.
    ///
    /// # Arguments
    ///
    /// * `incident_field` - The incident acoustic pressure field.
    /// * `bubble_radius` - Radius of bubbles/particles.
    /// * `bubble_velocity` - Velocity of bubbles/particles.
    /// * `grid` - Simulation grid.
    /// * `medium` - Medium properties.
    /// * `frequency` - Acoustic frequency.
    fn compute_scattering(
        &mut self,
        incident_field: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        bubble_velocity: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        frequency: f64,
    );

    /// Returns a reference to the 3D array of the computed scattered field.
    fn scattered_field(&self) -> &Array3<f64>;
}
