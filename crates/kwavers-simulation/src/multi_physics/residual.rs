//! Residual metrics for coupled multi-physics convergence.

use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::ArrayView3;

/// Compute the L-infinity update residual between two scalar fields.
///
/// The metric is
/// `||u_current - u_reference||_infinity = max_i |u_current[i] - u_reference[i]|`.
/// This is the contract used by fixed-point coupling: convergence requires every
/// coupled degree of freedom to satisfy the tolerance, not only the mean update.
///
/// The implementation is allocation-free: it streams both fields once and keeps
/// only the running maximum, giving O(n) time and O(1) auxiliary memory.
///
/// # Errors
///
/// Returns [`KwaversError::DimensionMismatch`] when field shapes differ and
/// [`KwaversError::InvalidInput`] when a field pair produces a non-finite update.
pub(super) fn max_abs_difference(
    current: ArrayView3<'_, f64>,
    reference: ArrayView3<'_, f64>,
) -> KwaversResult<f64> {
    let current_shape = current.dim();
    let reference_shape = reference.dim();
    if current_shape != reference_shape {
        return Err(KwaversError::DimensionMismatch(format!(
            "residual metric shape mismatch: current {current_shape:?}, reference {reference_shape:?}"
        )));
    }

    let mut residual = 0.0_f64;
    for (index, (&current_value, &reference_value)) in
        current.iter().zip(reference.iter()).enumerate()
    {
        let update = (current_value - reference_value).abs();
        if !update.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "non-finite residual update at linear index {index}: current={current_value}, reference={reference_value}"
            )));
        }
        residual = residual.max(update);
    }

    Ok(residual)
}

#[cfg(test)]
mod tests {
    use super::max_abs_difference;
    use kwavers_core::error::KwaversError;
    use ndarray::array;

    #[test]
    fn max_abs_difference_returns_l_infinity_norm() {
        let current = array![[[0.0, 2.0], [-1.0, 7.0]]];
        let reference = array![[[0.0, 1.0], [3.0, 1.0]]];

        let residual = max_abs_difference(current.view(), reference.view()).unwrap();

        assert_eq!(residual, 6.0);
    }

    #[test]
    fn max_abs_difference_rejects_shape_mismatch() {
        let current = array![[[1.0, 2.0]]];
        let reference = array![[[1.0], [2.0]]];

        let error = max_abs_difference(current.view(), reference.view()).unwrap_err();

        assert!(matches!(
            error,
            KwaversError::DimensionMismatch(message)
                if message.contains("current (1, 1, 2), reference (1, 2, 1)")
        ));
    }

    #[test]
    fn max_abs_difference_rejects_non_finite_update() {
        let current = array![[[f64::NAN]]];
        let reference = array![[[1.0]]];

        let error = max_abs_difference(current.view(), reference.view()).unwrap_err();

        assert!(matches!(
            error,
            KwaversError::InvalidInput(message)
                if message.contains("non-finite residual update")
        ));
    }
}
