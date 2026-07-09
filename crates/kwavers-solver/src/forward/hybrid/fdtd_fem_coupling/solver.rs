use super::config::FdtdFemCouplingConfig;
use super::coupler::FdtdFemCoupler;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_mesh::tetrahedral::TetrahedralMesh;
use leto::{
    Array3,
    ArrayView3,
};

/// FDTD-FEM Coupled Solver
#[derive(Debug)]
pub struct FdtdFemSolver {
    config: FdtdFemCouplingConfig,
    coupler: FdtdFemCoupler,
    fdtd_grid: Grid,
    fem_mesh: TetrahedralMesh,
    fdtd_field: Array3<f64>,
    fem_field: Vec<f64>,
    time_step: usize,
}

impl FdtdFemSolver {
    /// Create new coupled FDTD-FEM solver
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        config: FdtdFemCouplingConfig,
        fdtd_grid: Grid,
        fem_mesh: TetrahedralMesh,
    ) -> KwaversResult<Self> {
        let coupler = FdtdFemCoupler::new(config.clone(), &fdtd_grid, &fem_mesh)?;
        let fdtd_field = Array3::zeros((fdtd_grid.nx, fdtd_grid.ny, fdtd_grid.nz));
        let fem_field = vec![0.0; (fem_mesh.nodes.shape()[0] * fem_mesh.nodes.shape()[1] * fem_mesh.nodes.shape()[2])];

        Ok(Self {
            config,
            coupler,
            fdtd_grid,
            fem_mesh,
            fdtd_field,
            fem_field,
            time_step: 0,
        })
    }

    /// Perform coupled time step
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn step(&mut self) -> KwaversResult<()> {
        self.coupler.reset_convergence();

        for iteration in 0..self.config.max_iterations {
            let residual = self.coupler.schwarz_iteration(
                &mut self.fdtd_field,
                &mut self.fem_field,
                &self.fdtd_grid,
                &self.fem_mesh,
                iteration,
            )?;

            if residual < self.config.tolerance {
                log::debug!(
                    "Schwarz iteration converged after {} iterations (residual: {:.2e})",
                    iteration + 1,
                    residual
                );
                break;
            }

            if iteration == self.config.max_iterations - 1 {
                log::warn!(
                    "Schwarz iteration did not converge after {} iterations (residual: {:.2e})",
                    self.config.max_iterations,
                    residual
                );
            }
        }

        self.time_step += 1;
        Ok(())
    }

    /// Get current FDTD field
    pub fn fdtd_field(&self) -> ArrayView3<'_, f64> {
        self.fdtd_field.view()
    }

    /// Get current FEM field
    pub fn fem_field(&self) -> &[f64] {
        &self.fem_field
    }

    /// Get convergence history
    pub fn convergence_history(&self) -> &[f64] {
        self.coupler.convergence_history()
    }
}
