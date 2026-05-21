use super::super::config::FemPreconditionerType;
use super::FemHelmholtzSolver;
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::sparse::solver::{
    IterativeSolver, SolverConfig, SparsePreconditioner,
};

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
        let x0 = if self.solution.iter().any(|c| c.norm() > 0.0) {
            Some(self.solution.view())
        } else {
            None
        };

        self.solution = solver.bicgstab_complex(&self.system_matrix, self.rhs.view(), x0)?;
        Ok(())
    }
}
