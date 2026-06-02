//! Read-only accessors for the cached grid and material-property fields.

use ndarray::Array3;

use super::DiffusionSolver;
use kwavers_domain::grid::Grid;

impl DiffusionSolver {
    /// Get grid reference.
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Get diffusion coefficient field `D(r) = 1/(3(μₐ + μₛ'))`.
    pub fn diffusion_coefficient(&self) -> &Array3<f64> {
        &self.diffusion_coefficient
    }

    /// Get absorption coefficient field `μₐ(r)`.
    pub fn absorption_coefficient(&self) -> &Array3<f64> {
        &self.absorption_coefficient
    }
}
