//! Poroelastic solver for time-domain simulations
//!
//! Reference: Fellah & Depollier (2000) "Transient acoustic wave propagation"

use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_physics::acoustics::mechanics::poroelastic::{BiotTheory, PoroelasticMaterial};

/// Time-domain solver for poroelastic wave equations
#[derive(Debug)]
pub struct PoroelasticSolver {
    grid: Grid,
    biot: BiotTheory,
}

impl PoroelasticSolver {
    /// Create new poroelastic solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: &Grid, material: &PoroelasticMaterial) -> KwaversResult<Self> {
        let biot = BiotTheory::new(material);

        Ok(Self {
            grid: grid.clone(),
            biot,
        })
    }

    /// Time step for explicit scheme
    ///
    /// CFL condition: Δt ≤ Δx / c_max
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn compute_stable_timestep(&self) -> KwaversResult<f64> {
        let speeds = self.biot.compute_wave_speeds(MHZ_TO_HZ)?; // At 1 MHz
        let c_max = speeds.fast_wave;

        let dx_min = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let dt = 0.5 * dx_min / c_max; // Safety factor of 0.5

        Ok(dt)
    }
}
