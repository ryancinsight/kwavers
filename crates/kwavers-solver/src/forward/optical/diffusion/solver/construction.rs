//! `DiffusionSolver::new` / `DiffusionSolver::uniform` constructors and the
//! shared boundary-condition / ghost-coefficient helpers used by the
//! operator and preconditioner kernels.

use anyhow::Result;
use leto::Array3;

use super::{
    DiffusionBoundaryCondition, DiffusionBoundaryConditions, DiffusionSolver, DiffusionSolverConfig,
};
use kwavers_grid::Grid;
use kwavers_medium::properties::OpticalPropertyData;

impl DiffusionSolver {
    /// Resolve the boundary-condition record (configured override or
    /// extrapolated default keyed off `boundary_parameter`).
    pub(super) fn boundary_conditions(&self) -> DiffusionBoundaryConditions {
        self.config.boundary_conditions.unwrap_or_else(|| {
            DiffusionBoundaryConditions::all_extrapolated(self.config.boundary_parameter)
        })
    }

    /// Compute the ghost-cell scaling factor for the extrapolated-boundary
    /// closure: `Φ_{ghost} = ghost_coefficient · Φ_{boundary}`.
    pub(super) fn ghost_coefficient(
        boundary_condition: DiffusionBoundaryCondition,
        diffusion_coefficient: f64,
        delta: f64,
    ) -> f64 {
        match boundary_condition {
            DiffusionBoundaryCondition::ZeroFlux => 1.0,
            DiffusionBoundaryCondition::Extrapolated { a } => {
                let r = 4.0 * a * diffusion_coefficient / delta;
                if r <= 0.0 {
                    0.0
                } else if (r + 1.0).abs() > 1e-30 {
                    (r - 1.0) / (r + 1.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Create solver from spatially-varying optical property map.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        grid: Grid,
        optical_properties: Array3<OpticalPropertyData>,
        config: DiffusionSolverConfig,
    ) -> Result<Self> {
        let (nx, ny, nz) = grid.dimensions();

        if optical_properties.shape() != [nx, ny, nz] {
            anyhow::bail!(
                "Optical property map dimensions {:?} do not match grid dimensions ({}, {}, {})",
                optical_properties.shape(),
                nx,
                ny,
                nz
            );
        }

        let mut diffusion_coefficient = Array3::zeros((nx, ny, nz));
        let mut absorption_coefficient = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let props = &optical_properties[[i, j, k]];

                    let mu_a = props.absorption_coefficient;
                    let mu_s_prime = props.reduced_scattering();
                    let d_val = 1.0 / (3.0 * (mu_a + mu_s_prime));

                    diffusion_coefficient[[i, j, k]] = d_val;
                    absorption_coefficient[[i, j, k]] = mu_a;
                }
            }
        }

        Ok(Self {
            grid,
            diffusion_coefficient,
            absorption_coefficient,
            config,
        })
    }

    /// Create solver with uniform optical properties (homogeneous medium).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn uniform(
        grid: Grid,
        optical_properties: OpticalPropertyData,
        config: DiffusionSolverConfig,
    ) -> Result<Self> {
        let (nx, ny, nz) = grid.dimensions();
        let optical_map = Array3::from_elem([nx, ny, nz], optical_properties);
        Self::new(grid, optical_map, config)
    }
}
