//! Utility functions for adaptive beamforming algorithms
//!
//! # SSOT enforcement
//! This module previously contained ad-hoc complex matrix inversion and Hermitian eigensolvers.
//! Those implementations duplicated numerics already owned by the SSOT linear algebra layer and
//! introduced silent failure modes (returning `None` / `0.0` fallbacks), which violates the
//! project mandate of explicit invariants and no error masking.
//!
//! **Policy:** Adaptive algorithms must route all linear algebra through
//! `crate::utils::linear_algebra::LinearAlgebra` (solve / inversion) and must surface failures via
//! `KwaversResult` rather than fallback outputs.

use crate::error::{KwaversError, KwaversResult, NumericalError};

/// `adaptive::algorithms` no longer provides local numerics.
///
/// This function exists only to produce a clear compile-time error if legacy callers attempt
/// to use this module for inversion/eigendecomposition.
///
/// # Errors
/// Always returns `Err`.
#[inline]
pub(super) fn ssot_only() -> KwaversResult<()> {
    Err(KwaversError::Numerical(NumericalError::UnsupportedOperation {
        operation: "adaptive::algorithms::utils (local numerics)".to_string(),
        reason: "Duplicated numerics were removed. Use SSOT `crate::utils::linear_algebra::LinearAlgebra`.".to_string(),
    }))
}
