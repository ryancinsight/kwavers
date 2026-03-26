use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;
use std::fmt::Debug;

/// Trait for chemical models.
///
/// Implementors of this trait simulate chemical reactions, radical formation,
/// and other chemical processes occurring within the medium.
pub trait ChemicalModelTrait: Debug + Send + Sync {
    /// Advances the chemical simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `p` - Acoustic pressure field.
    /// * `light` - Light intensity field.
    /// * `emission_spectrum` - Emission spectrum from other processes (e.g., sonoluminescence).
    /// * `bubble_radius` - Field of bubble radii.
    /// * `temperature` - Temperature field.
    /// * `grid` - Simulation grid.
    /// * `dt` - Time step.
    /// * `medium` - Medium properties.
    /// * `frequency` - Acoustic frequency.
    #[allow(clippy::too_many_arguments)]
    fn update_chemical(
        &mut self,
        p: &Array3<f64>,
        light: &Array3<f64>,
        emission_spectrum: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        temperature: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    );

    /// Returns a reference to the 3D array of the primary radical concentration.
    fn radical_concentration(&self) -> &Array3<f64>;
}
