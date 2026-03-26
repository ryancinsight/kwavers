use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, Array4};
use std::fmt::Debug;

/// Trait for light diffusion models.
///
/// Implementors of this trait simulate the diffusion of light through a medium,
/// considering sources, absorption, scattering, and potentially other optical effects.
pub trait LightDiffusionModelTrait: Debug + Send + Sync {
    /// Advances the light diffusion simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `fields` - A mutable reference to a 4D array representing the simulation fields.
    ///   The light field (e.g., fluence rate) is updated in place.
    /// * `light_source` - A 3D array representing the light source term (e.g., from sonoluminescence).
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`, providing optical properties.
    /// * `dt` - The time step size for this update.
    fn update_light(
        &mut self,
        fields: &mut Array4<f64>,
        light_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    );

    /// Returns a reference to the 3D array of the emission spectrum.
    fn emission_spectrum(&self) -> &Array3<f64>;

    /// Returns a reference to the 4D array of the fluence rate.
    /// The first dimension typically represents different wavelengths or components if applicable,
    /// or is singular if monochromatic.
    fn fluence_rate(&self) -> &Array4<f64>;

    /// Reports performance metrics of the light diffusion model.
    fn report_performance(&self);
}
