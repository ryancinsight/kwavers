use super::{coupler::PstdSemCoupler, PstdSemCouplingConfig};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use ndarray::{Array3, ArrayView3};

/// PSTD-SEM Coupled Solver
#[derive(Debug)]
pub struct PstdSemSolver {
    config: PstdSemCouplingConfig,
    coupler: PstdSemCoupler,
    pstd_grid: Grid,
    sem_mesh: TetrahedralMesh,
    pstd_field: Array3<f64>,
    sem_field: Vec<f64>,
}

impl PstdSemSolver {
    /// Create new coupled PSTD-SEM solver
    pub fn new(
        config: PstdSemCouplingConfig,
        pstd_grid: Grid,
        sem_mesh: TetrahedralMesh,
    ) -> KwaversResult<Self> {
        let coupler = PstdSemCoupler::new(config.clone(), &pstd_grid, &sem_mesh)?;
        let pstd_field = Array3::zeros((pstd_grid.nx, pstd_grid.ny, pstd_grid.nz));
        let sem_field = vec![0.0; sem_mesh.nodes.len()];

        Ok(Self {
            config,
            coupler,
            pstd_grid,
            sem_mesh,
            pstd_field,
            sem_field,
        })
    }

    /// Perform coupled time step
    pub fn step(&mut self) -> KwaversResult<f64> {
        self.coupler.reset_convergence();

        let residual = self.coupler.couple_fields(
            &mut self.pstd_field,
            &mut self.sem_field,
            &self.pstd_grid,
            &self.sem_mesh,
        )?;

        Ok(residual)
    }

    /// Get current PSTD field
    pub fn pstd_field(&self) -> ArrayView3<'_, f64> {
        self.pstd_field.view()
    }

    /// Get current SEM field
    pub fn sem_field(&self) -> &[f64] {
        &self.sem_field
    }

    /// Get convergence history
    pub fn convergence_history(&self) -> &[f64] {
        self.coupler.convergence_history()
    }

    /// Get coupling configuration
    pub fn config(&self) -> &PstdSemCouplingConfig {
        &self.config
    }
}
