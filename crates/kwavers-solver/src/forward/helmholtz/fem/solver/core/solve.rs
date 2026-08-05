use super::super::config::FemPreconditionerType;
use super::FemHelmholtzSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::linear_algebra::sparse::solver::{
    IterativeSolver, SolverConfig, SparsePreconditioner,
};
use leto::Array1;

impl FemHelmholtzSolver {
    /// Solve the assembled system via BiCGSTAB with the configured preconditioner.
    /// # Errors
    /// - Returns [`crate::KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
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
            Some(self.solution.as_slice().expect("solution must be contiguous"))
        } else {
            None
        };

        let result = solver.bicgstab_complex(&self.system_matrix, self.rhs.as_slice().expect("rhs must be contiguous"), x0)?;
        self.solution = Array1::from_vec([result.len()], result)?;
        Ok(())
    }
}
