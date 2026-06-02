//! Residual norm metrics for Newton-Krylov convergence tests.

use ndarray::Array3;

/// Compute squared L2 norm without taking a square root.
///
/// Line-search candidate comparison only needs norm ordering.  Squared norms
/// preserve that order for nonnegative norms and avoid one `sqrt` per trial.
pub(in crate::multiphysics::monolithic) fn norm_squared(a: &Array3<f64>) -> f64 {
    a.iter().fold(0.0, |sum, &value| value.mul_add(value, sum))
}

/// Compute L2 norm.
pub(in crate::multiphysics::monolithic) fn norm(a: &Array3<f64>) -> f64 {
    norm_squared(a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_norm_squared_matches_l2_norm_contract() {
        let field = Array3::from_shape_vec((2, 2, 1), vec![3.0, 4.0, 12.0, 0.0]).unwrap();

        assert_eq!(norm_squared(&field), 169.0);
        assert_eq!(norm(&field), 13.0);
    }
}
