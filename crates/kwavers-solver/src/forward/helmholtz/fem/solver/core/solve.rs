use super::super::config::FemPreconditionerType;
use super::FemHelmholtzSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::linear_algebra::sparse::solver::{
    IterativeSolver, SolverConfig, SparsePreconditioner,
};
use leto::Array1 as LetoArray1;

impl FemHelmholtzSolver {
    /// Solve the assembled system via BiCGSTAB with the configured preconditioner.
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn solve_system(&mut self) -> KwaversResult<()> {
        let preconditioner = match self.config.preconditioner {
            FemPreconditionerType::None => SparsePreconditioner::None,
            FemPreconditionerType::Diagonal => SparsePreconditioner::Jacobi,
            FemPreconditionerType::ILU => {
                return Err(KwaversError::FeatureNotAvailable(
                    "FEM Helmholtz ILU preconditioner requires a real sparse incomplete factorization backend".to_owned(),
                ))
            }
            FemPreconditionerType::AMG => {
                return Err(KwaversError::FeatureNotAvailable(
                    "FEM Helmholtz AMG preconditioner requires a real multigrid hierarchy backend".to_owned(),
                ))
            }
        };

        let config = SolverConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner,
            verbose: false,
        };

        let solver = IterativeSolver::create(config);
        let rhs_leto = LetoArray1::from_shape_fn([self.rhs.len()], |[i]| self.rhs[i]);
        let x0 = if self.solution.iter().any(|c| c.norm() > 0.0) {
            Some(LetoArray1::from_shape_fn([self.solution.len()], |[i]| {
                self.solution[i]
            }))
        } else {
            None
        };

        let solved = solver.bicgstab_complex(&self.system_matrix, &rhs_leto, x0.as_ref())?;
        for i in 0..self.solution.len() {
            self.solution[i] = solved[[i]];
        }
        Ok(())
    }
}
