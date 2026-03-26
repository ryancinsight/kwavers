use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;
use std::fmt::Debug;

/// Trait for cavitation models.
///
/// Implementors of this trait simulate bubble dynamics, considering the nonlinear
/// effects on the acoustic field and other physical phenomena (e.g., sonoluminescence).
pub trait CavitationModelBehavior: Debug + Send + Sync {
    /// Advances the cavitation simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `pressure` - A reference to the 3D acoustic pressure field driving bubble dynamics.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`.
    /// * `dt` - The time step size for this update.
    /// * `t` - The current simulation time.
    ///
    /// # Returns
    ///
    /// A `KwaversResult<()>` indicating success or failure of the update.
    fn update_cavitation(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> crate::core::error::KwaversResult<()>;

    /// Returns the 3D array of bubble radii (meters).
    fn bubble_radius(&self) -> crate::core::error::KwaversResult<Array3<f64>>;

    /// Returns the 3D array of bubble wall velocities (m/s).
    fn bubble_velocity(&self) -> crate::core::error::KwaversResult<Array3<f64>>;

    /// Returns the 3D array of light emission from sonoluminescence (W/m³).
    fn light_emission(&self) -> Array3<f64>;

    /// Reports performance metrics of the cavitation model.
    fn report_performance(&self);
}
