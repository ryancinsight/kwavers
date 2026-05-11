//! Acoustic adjoint-state primitives.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{s, Array2, Array3, ArrayView3, Zip};

fn validate_pair_shapes(
    observed: &Array2<f64>,
    synthetic: &Array2<f64>,
) -> KwaversResult<(usize, usize)> {
    if observed.dim() != synthetic.dim() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "Observed and synthetic data shape mismatch: observed {:?}, synthetic {:?}",
                    observed.dim(),
                    synthetic.dim()
                ),
            },
        ));
    }

    Ok(observed.dim())
}

/// Compute the discrete L2 residual.
///
/// The residual is defined as `d_syn - d_obs`, which is the gradient of
/// `1/2 ||d_syn - d_obs||²` with respect to the synthetic data.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn l2_residual(observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<Array2<f64>> {
    validate_pair_shapes(observed, synthetic)?;
    Ok(synthetic - observed)
}

/// Compute the discrete L2 objective.
///
/// ## Theorem
/// For `J = (dt / 2) ||d_syn - d_obs||²`, the objective is non-negative and
/// vanishes if and only if `d_syn = d_obs` pointwise.
/// # Errors
/// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn l2_objective(
    dt: f64,
    observed: &Array2<f64>,
    synthetic: &Array2<f64>,
) -> KwaversResult<f64> {
    if dt <= 0.0 || !dt.is_finite() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: "Objective timestep must be positive and finite".to_owned(),
            },
        ));
    }

    validate_pair_shapes(observed, synthetic)?;
    let residual = synthetic - observed;
    Ok(0.5 * dt * residual.mapv(|x| x * x).sum())
}

/// Reverse the sample axis of a trace matrix.
///
/// The input is interpreted as `(receiver, time)`. The output is an exact
/// reversal of the time axis.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
#[must_use] 
pub fn reverse_time_axis(data: &Array2<f64>) -> Array2<f64> {
    data.slice(s![.., ..;-1]).to_owned()
}

/// Accumulate a signed zero-lag correlation into a gradient volume.
///
/// ## Theorem
/// If the discrete gradient is a weighted sum of pointwise products over
/// matching time slices, then `G += scale * forward ⊙ adjoint` is the exact
/// discrete imaging-condition update.
/// # Errors
/// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn accumulate_signed_correlation(
    gradient: &mut Array3<f64>,
    forward: ArrayView3<'_, f64>,
    adjoint: ArrayView3<'_, f64>,
    scale: f64,
) -> KwaversResult<()> {
    if gradient.dim() != forward.dim() || forward.dim() != adjoint.dim() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "Signed correlation shape mismatch: gradient {:?}, forward {:?}, adjoint {:?}",
                    gradient.dim(),
                    forward.dim(),
                    adjoint.dim()
                ),
            },
        ));
    }

    Zip::from(gradient.view_mut())
        .and(forward)
        .and(adjoint)
        .par_for_each(|g, &f, &a| {
            *g += scale * f * a;
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_l2_residual_is_synthetic_minus_observed() {
        let observed = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).expect("shape");
        let synthetic = Array2::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).expect("shape");

        let residual = l2_residual(&observed, &synthetic).expect("residual");
        assert_eq!(
            residual,
            Array2::from_shape_vec((1, 3), vec![3.0, 3.0, 3.0]).expect("shape")
        );
    }

    #[test]
    fn test_l2_objective_scales_with_dt() {
        let observed = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("shape");
        let synthetic = Array2::from_shape_vec((1, 2), vec![3.0, 5.0]).expect("shape");

        let objective = l2_objective(0.5, &observed, &synthetic).expect("objective");
        assert!((objective - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reverse_time_axis() {
        let data =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
        let reversed = reverse_time_axis(&data);
        assert_eq!(
            reversed,
            Array2::from_shape_vec((2, 3), vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]).expect("shape")
        );
    }

    #[test]
    fn test_accumulate_signed_correlation_matches_manual_sum() {
        let mut gradient = Array3::zeros((2, 2, 2));
        let forward = Array3::from_elem((2, 2, 2), 2.0);
        let adjoint = Array3::from_elem((2, 2, 2), 3.0);

        accumulate_signed_correlation(&mut gradient, forward.view(), adjoint.view(), -0.5)
            .expect("accumulation");

        assert!(gradient.iter().all(|&v| (v + 3.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_accumulate_signed_correlation_rejects_shape_mismatch() {
        let mut gradient = Array3::zeros((2, 2, 2));
        let forward = Array3::from_elem((2, 2, 2), 2.0);
        let adjoint = Array3::from_elem((3, 2, 2), 3.0);

        let err = accumulate_signed_correlation(&mut gradient, forward.view(), adjoint.view(), 1.0)
            .expect_err("shape mismatch must fail");

        assert!(format!("{err:?}").contains("Signed correlation shape mismatch"));
    }
}
