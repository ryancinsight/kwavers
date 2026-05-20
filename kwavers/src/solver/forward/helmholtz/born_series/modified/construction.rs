use crate::core::constants::cavitation::VISCOSITY_WATER;
use crate::core::constants::thermodynamic::{SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;
use num_complex::Complex64;

use super::{
    super::{BornConfig, BornWorkspace},
    ModifiedBornSolver,
};

impl ModifiedBornSolver {
    pub fn new(config: BornConfig, grid: Grid) -> Self {
        let workspace = BornWorkspace::new(grid.nx, grid.ny, grid.nz);
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            config,
            grid,
            workspace,
            absorption_field: Array3::zeros(shape),
            diffusivity_field: Array3::zeros(shape),
        }
    }
    /// Precompute viscoacoustic properties.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn precompute_viscoacoustic_properties<M: Medium>(
        &mut self,
        frequency: f64,
        medium: &M,
    ) -> KwaversResult<()> {
        let omega = 2.0 * std::f64::consts::PI * frequency;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let c0 = medium.sound_speed(i, j, k);
                    let diffusivity = self.compute_diffusivity(medium, i, j, k);

                    self.diffusivity_field[[i, j, k]] = diffusivity;

                    let absorption = (omega * omega * diffusivity) / (2.0 * c0 * c0 * c0);
                    self.absorption_field[[i, j, k]] = Complex64::new(0.0, absorption);
                }
            }
        }

        Ok(())
    }

    pub(super) fn compute_diffusivity<M: Medium>(
        &self,
        medium: &M,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let rho = medium.density(i, j, k);
        // Water-coupling baseline (sourced from SSOT — see core::constants).
        let viscosity = VISCOSITY_WATER;
        let thermal_conductivity = THERMAL_CONDUCTIVITY_WATER;
        let heat_capacity = SPECIFIC_HEAT_WATER;

        let viscous_diffusivity = (4.0 / 3.0) * viscosity / rho;
        let thermal_diffusivity = thermal_conductivity / (rho * heat_capacity);

        viscous_diffusivity + thermal_diffusivity
    }
}
