//! Monte Carlo photon transport solver facade.

mod geometry;
mod simulation;
mod trace;

use kwavers_grid::Grid3D;
use kwavers_medium::optical_map::OpticalPropertyMap;

/// Monte Carlo photon transport solver.
#[derive(Debug)]
pub struct MonteCarloSolver {
    pub(super) grid: Grid3D,
    pub(super) optical_map: OpticalPropertyMap,
}

impl MonteCarloSolver {
    /// Create a Monte Carlo solver over a grid and optical property map.
    #[must_use]
    pub fn new(grid: Grid3D, optical_map: OpticalPropertyMap) -> Self {
        Self { grid, optical_map }
    }
}
