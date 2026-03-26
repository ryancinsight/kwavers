use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use ndarray::{Array3, Array4};
use std::fmt::Debug;

/// Trait for acoustic wave propagation models.
///
/// Implementors of this trait are responsible for updating the acoustic wave field
/// over a time step `dt`, considering nonlinear effects, source terms, and medium properties.
pub trait AcousticWaveModel: Debug + Send + Sync {
    /// Advances the acoustic wave simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `fields` - A mutable reference to a 4D array representing the simulation fields.
    ///   The pressure field, typically at `fields.index_axis(Axis(0), PRESSURE_IDX)`, is updated in place.
    /// * `prev_pressure` - A reference to the 3D pressure field from the previous time step.
    /// * `source` - A trait object implementing `Source`, defining the acoustic source.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`, providing material properties.
    /// * `dt` - The time step size for this update.
    /// * `t` - The current simulation time.
    #[allow(clippy::too_many_arguments)]
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()>;

    /// Reports performance metrics of the wave model.
    ///
    /// This method should log or print information about the computational performance,
    /// such as time spent in different parts of the simulation or number of calls.
    fn report_performance(&self);

    /// Sets the scaling factor for the nonlinearity term.
    fn set_nonlinearity_scaling(&mut self, scaling: f64);

    // Consider adding methods for configuration if common settings are identifiable.
    // Adaptive timestep configuration methods are implementation-specific and vary
    // significantly between FDTD, PSTD, and DG solvers. Each implementation provides
    // its own configuration API rather than enforcing a common interface here.
}
