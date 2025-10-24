//! Poroelastic solver for time-domain simulations
//!
//! Reference: Fellah & Depollier (2000) "Transient acoustic wave propagation"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::mechanics::poroelastic::{BiotTheory, PoroelasticMaterial};
use ndarray::Array3;

/// Time-domain solver for poroelastic wave equations
#[derive(Debug)]
pub struct PoroelasticSolver {
    grid: Grid,
    #[allow(dead_code)]
    material: PoroelasticMaterial,
    biot: BiotTheory,
}

impl PoroelasticSolver {
    /// Create new poroelastic solver
    pub fn new(grid: &Grid, material: &PoroelasticMaterial) -> KwaversResult<Self> {
        let biot = BiotTheory::new(material);

        Ok(Self {
            grid: grid.clone(),
            material: material.clone(),
            biot,
        })
    }

    /// Time step for explicit scheme
    ///
    /// CFL condition: Δt ≤ Δx / c_max
    pub fn compute_stable_timestep(&self) -> KwaversResult<f64> {
        let speeds = self.biot.compute_wave_speeds(1e6)?; // At 1 MHz
        let c_max = speeds.fast_wave;

        let dx_min = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let dt = 0.5 * dx_min / c_max; // Safety factor of 0.5

        Ok(dt)
    }

    /// Step forward in time (stub for full implementation)
    pub fn step(
        &self,
        solid_displacement: &Array3<f64>,
        fluid_displacement: &Array3<f64>,
        _dt: f64,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        // This would implement the full Biot equations
        // For now, return unchanged (production implementation would solve PDEs)
        Ok((solid_displacement.clone(), fluid_displacement.clone()))
    }
}
