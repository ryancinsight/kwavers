//! Convergence summary read out of an Athena solve report.

use athena_core::SolveReport;
use kwavers_core::error::{KwaversError, NumericalError};

/// Convergence information for a GMRES solve.
#[derive(Debug, Clone)]
pub struct GmresConvergenceInfo {
    /// Whether the solve met its convergence policy.
    pub converged: bool,
    /// Krylov iterations performed.
    pub iterations: usize,
    /// Final residual norm ‖b − A·x‖.
    pub final_residual: f64,
    /// Residual reduction ‖r‖ / ‖r₀‖ achieved over the solve.
    ///
    /// Every kwavers call site starts from the zero iterate, where `‖r₀‖ = ‖b‖`
    /// and this is the relative residual against the right-hand side.
    pub relative_residual: f64,
}

impl GmresConvergenceInfo {
    /// Summarise an Athena solve report.
    #[must_use]
    pub(crate) fn from_report(report: &SolveReport<f64>) -> Self {
        Self {
            converged: report.converged(),
            iterations: report.iterations,
            final_residual: report.final_residual_norm,
            relative_residual: relative_residual(report),
        }
    }
}

/// Residual reduction achieved by a solve.
///
/// A zero initial residual means the initial iterate already solved the system,
/// not a division to guard: no residual remains, so the reduction is zero.
fn relative_residual(report: &SolveReport<f64>) -> f64 {
    if report.initial_residual_norm > 0.0 {
        report.final_residual_norm / report.initial_residual_norm
    } else {
        0.0
    }
}

/// Report a solve that did not meet its convergence policy as a typed failure.
///
/// Athena reports a stalled, stagnated, or broken-down solve value-semantically
/// in its [`SolveReport`] rather than as an error, because which terminal
/// condition was reached is information the caller branches on. Call sites
/// whose own contract is "converged or fail" convert it here, keeping the
/// terminal condition in the message.
pub(crate) fn convergence_failure(method: &str, report: &SolveReport<f64>) -> KwaversError {
    KwaversError::Numerical(NumericalError::ConvergenceFailed {
        method: format!("{method} ({:?})", report.termination),
        iterations: report.iterations,
        error: relative_residual(report),
    })
}
