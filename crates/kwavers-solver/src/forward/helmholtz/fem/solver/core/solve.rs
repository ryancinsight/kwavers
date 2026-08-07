use super::super::config::FemPreconditionerType;
use super::FemHelmholtzSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::complex_solve;

impl FemHelmholtzSolver {
    /// Solve the assembled system through Leto's provider-owned complex solver.
    /// # Errors
    /// - Returns [`crate::KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn solve_system(&mut self) -> KwaversResult<()> {
        match self.config.preconditioner {
            FemPreconditionerType::None | FemPreconditionerType::Diagonal => {}
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
        }

        let matrix = self.system_matrix.to_dense_array()?;
        self.solution = complex_solve(&matrix, &self.rhs)?;
        Ok(())
    }
}
