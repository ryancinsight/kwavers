use super::{BemFemCoupler, BemFemCouplingConfig, BemFemInterface};
use crate::core::error::KwaversResult;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use num_complex::Complex64;

/// BEM-FEM Coupled Solver for Helmholtz problems
#[derive(Debug)]
pub struct BemFemSolver {
    config: BemFemCouplingConfig,
    coupler: BemFemCoupler,
    fem_mesh: TetrahedralMesh,
    _bem_boundary_elements: Vec<usize>,
    wavenumber: f64,
}

impl BemFemSolver {
    /// Create new BEM-FEM coupled solver
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        config: BemFemCouplingConfig,
        fem_mesh: TetrahedralMesh,
        bem_boundary_elements: Vec<usize>,
        wavenumber: f64,
    ) -> KwaversResult<Self> {
        let coupler = BemFemCoupler::new(config.clone(), &fem_mesh, &bem_boundary_elements)?;

        Ok(Self {
            config,
            coupler,
            fem_mesh,
            _bem_boundary_elements: bem_boundary_elements,
            wavenumber,
        })
    }

    /// Solve the coupled BEM-FEM system
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn solve(
        &mut self,
        mut fem_initial_guess: Vec<Complex64>,
        mut bem_boundary_guess: Vec<Complex64>,
    ) -> KwaversResult<()> {
        self.coupler.solve_coupled(
            &mut fem_initial_guess,
            &mut bem_boundary_guess,
            &self.fem_mesh,
            self.wavenumber,
        )?;

        Ok(())
    }

    /// Get the coupling interface
    #[must_use] 
    pub fn interface(&self) -> &BemFemInterface {
        &self.coupler.interface
    }

    /// Get convergence information
    #[must_use] 
    pub fn convergence_info(&self) -> (bool, usize, &[f64]) {
        (
            self.coupler.has_converged(),
            self.coupler.iterations(),
            self.coupler.convergence_history(),
        )
    }

    /// Get configuration
    #[must_use] 
    pub fn config(&self) -> &BemFemCouplingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::tetrahedral::BoundaryType;

    #[test]
    fn test_bem_fem_solver_creation() {
        let mut fem_mesh = TetrahedralMesh::new();
        fem_mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);

        let bem_boundary = vec![0];
        let config = BemFemCouplingConfig::default();
        let wavenumber = 2.0 * std::f64::consts::PI * 1e6 / 1482.0; // k = 2πf/c

        let solver = BemFemSolver::new(config, fem_mesh, bem_boundary, wavenumber);

        let _solver = solver.unwrap();
    }
}
