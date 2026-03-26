use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, Array4};
use std::fmt::Debug;

/// Trait for thermal models.
///
/// Implementors of this trait simulate heat transfer and temperature changes within the medium,
/// considering heat sources (e.g., acoustic, optical) and thermal diffusion.
pub trait ThermalModelTrait: Debug + Send + Sync {
    /// Advances the thermal simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `fields` - A mutable reference to a 4D array representing the simulation fields.
    ///   The temperature field is updated in place.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`, providing thermal properties.
    /// * `dt` - The time step size for this update.
    /// * `frequency` - The acoustic frequency, relevant for some heat source calculations.
    fn update_thermal(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        frequency: f64,
    );

    /// Returns a reference to the 3D array of the current temperature field (Kelvin).
    fn temperature(&self) -> &Array3<f64>;

    /// Sets the 3D temperature field.
    /// Required for numerical solvers that manage state externally or for initialization.
    fn set_temperature(&mut self, temperature: &Array3<f64>);

    /// Reports performance metrics of the thermal model.
    fn report_performance(&self);
}
