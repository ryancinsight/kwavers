//! Shared helpers for velocity statistics modules.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array3};

/// Copy field values at sensor positions into caller-owned storage.
pub(super) fn fill_field_at_positions(
    field: &Array3<f64>,
    positions: &[(usize, usize, usize)],
    out: &mut Array1<f64>,
) -> KwaversResult<()> {
    validate_sample_output_len(positions, out)?;
    for (row, &(i, j, k)) in positions.iter().enumerate() {
        out[row] = field[[i, j, k]];
    }
    Ok(())
}

/// Verify that `out` has the same length as `positions`.
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
