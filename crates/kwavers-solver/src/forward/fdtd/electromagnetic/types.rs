//! ElectromagneticFdtdSolver struct definition.

use kwavers_field::EMFields;
use kwavers_grid::Grid;
use kwavers_physics::electromagnetic::equations::EMMaterialDistribution;
use ndarray::Array3;

/// Electromagnetic FDTD solver using Yee's algorithm
///
/// Solves Maxwell's equations:
/// ```text
/// ∂E/∂t = (1/ε) ∇ × H - (σ/ε) E    (Faraday's Law)
/// ∂H/∂t = -(1/μ) ∇ × E             (Ampere's Law)
/// ```
///
/// The Yee cell staggers E and H fields in both space and time for numerical stability.
pub struct ElectromagneticFdtdSolver {
    /// Computational grid
    pub(super) grid: Grid,
    /// Electromagnetic material properties (ε, μ, σ)
    pub(super) materials: EMMaterialDistribution,
    /// Current time step
    pub(super) time_step: usize,
    /// Time step size (seconds)
    pub(super) dt: f64,
    /// Electric field components on Yee grid
    pub(super) ex: Array3<f64>,
    pub(super) ey: Array3<f64>,
    pub(super) ez: Array3<f64>,
    /// Magnetic field components on Yee grid
    pub(super) hx: Array3<f64>,
    pub(super) hy: Array3<f64>,
    pub(super) hz: Array3<f64>,
    /// Cell-centered electromagnetic fields (derived from Yee grid)
    pub(super) fields_cache: EMFields,
}

impl std::fmt::Debug for ElectromagneticFdtdSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElectromagneticFdtdSolver")
            .field("grid", &self.grid)
            .field("materials", &self.materials)
            .field("time_step", &self.time_step)
            .field("dt", &self.dt)
            .finish_non_exhaustive()
    }
}
