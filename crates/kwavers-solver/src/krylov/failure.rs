//! Athena solve failures in kwavers' typed vocabulary.
//!
//! Athena reports dimension and backend failures through
//! [`SolveError`](athena_core::SolveError) over the Leto backend; kwavers
//! carries them as [`KwaversError`]. These conversions are the one place the
//! translation happens, so every Krylov entry point in this crate — the GMRES
//! bridge, the CG entry, the boundary-element solver, the monolithic coupler —
//! maps failures identically.

use athena_core::SolveError;
use athena_leto::LetoBackendError;
use kwavers_core::error::{KwaversError, NumericalError};

/// Convert an Athena solve failure into kwavers' typed numerical vocabulary.
pub(crate) fn solve_failure(error: &SolveError<LetoBackendError>) -> KwaversError {
    match error {
        SolveError::DimensionMismatch {
            context,
            expected,
            actual,
        } => KwaversError::Numerical(NumericalError::MatrixDimension {
            operation: format!("GMRES {context}"),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }),
        SolveError::Backend(backend) => backend_failure("GMRES solve", backend),
        // `SolveError` is `#[non_exhaustive]`: a variant Athena adds later must
        // surface as a solver failure rather than silently taking a branch that
        // was written for a different condition.
        other => KwaversError::Numerical(NumericalError::SolverFailed {
            method: "GMRES solve".to_owned(),
            reason: other.to_string(),
        }),
    }
}

/// Convert a backend allocation or arithmetic failure into a typed error.
pub(crate) fn backend_failure(operation: &str, error: &LetoBackendError) -> KwaversError {
    KwaversError::Numerical(NumericalError::SolverFailed {
        method: operation.to_owned(),
        reason: error.to_string(),
    })
}
