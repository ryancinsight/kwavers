//! Shared helpers for velocity statistics modules.

use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array1;

/// Verify that `out` has the same length as `positions`.
/// # Errors
/// - Returns [`KwaversError::DimensionMismatch`] if the precondition for mismatched array or grid dimensions is violated.
///
pub(super) fn validate_sample_output_len(
    positions: &[(usize, usize, usize)],
    out: &Array1<f64>,
) -> KwaversResult<()> {
    if out.len() != positions.len() {
        return Err(KwaversError::DimensionMismatch(format!(
            "velocity-stat output length {} != sensor count {}",
            out.len(),
            positions.len()
        )));
    }
    Ok(())
}
