//! GMRES configuration in kwavers vocabulary.

use athena_core::ConvergencePolicy;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};

/// GMRES solver configuration.
///
/// The restart width and the tolerances are kwavers' vocabulary; the meaning
/// they carry is Athena's. A residual `r` is converged when
/// `‖r‖₂ <= max(absolute_tolerance, relative_tolerance · ‖b‖₂)`.
#[derive(Debug, Clone)]
pub struct GMRESConfig {
    /// Krylov subspace dimension before restart (typical: 10–100).
    pub krylov_dim: usize,
    /// Maximum number of outer restart cycles.
    pub max_iterations: usize,
    /// Relative tolerance: ‖r‖ / ‖b‖ < tol.
    pub relative_tolerance: f64,
    /// Absolute tolerance: ‖r‖ < tol.
    pub absolute_tolerance: f64,
}

impl Default for GMRESConfig {
    fn default() -> Self {
        Self {
            krylov_dim: 30,
            max_iterations: 100,
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-10,
        }
    }
}

impl GMRESConfig {
    /// Translate this configuration into a validated Athena policy.
    ///
    /// Athena budgets total Krylov iterations, while this configuration counts
    /// outer restart cycles of `krylov_dim` iterations each. The budget is
    /// therefore the product, which is the same operator-application count a
    /// restarted solve was allowed before.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalError::InvalidOperation`] when the iteration budget
    /// overflows `usize`, or when Athena rejects the tolerances or the budget
    /// as non-finite, negative, or zero.
    pub fn policy(&self) -> KwaversResult<ConvergencePolicy<f64>> {
        let budget = self
            .max_iterations
            .checked_mul(self.krylov_dim)
            .ok_or_else(|| {
                KwaversError::Numerical(NumericalError::InvalidOperation(format!(
                    "GMRES iteration budget {} restarts x {} Krylov dimensions overflows usize",
                    self.max_iterations, self.krylov_dim
                )))
            })?;
        policy(self.absolute_tolerance, self.relative_tolerance, budget)
    }
}

/// Build a validated Athena convergence policy from explicit tolerances.
///
/// # Errors
///
/// Returns [`NumericalError::InvalidOperation`] naming Athena's rejection
/// reason when a tolerance is non-finite or negative, or the budget is zero.
pub(crate) fn policy(
    absolute_tolerance: f64,
    relative_tolerance: f64,
    max_iterations: usize,
) -> KwaversResult<ConvergencePolicy<f64>> {
    ConvergencePolicy::new(absolute_tolerance, relative_tolerance, max_iterations).map_err(
        |reason| {
            KwaversError::Numerical(NumericalError::InvalidOperation(format!(
                "invalid GMRES convergence policy (absolute {absolute_tolerance:.3e}, \
                 relative {relative_tolerance:.3e}, budget {max_iterations}): {reason}"
            )))
        },
    )
}
