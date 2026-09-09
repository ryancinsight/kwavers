use super::helpers::{autodiff_second_derivative_xx, second_difference_xx};
use super::{FD_H_SECOND, REL_TOL_SECOND};
use crate::inverse::elastic_2d::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;

type B = super::TestBackend;

// Autodiff-vs-second-difference check on identically initialized models.
// Measured 2026-08-22: ~20 ms, stable across 5/5 runs (rel tol 1e-2,
// h = 1e-4), so the previous ignore is lifted.
//
// Both sides are finite-difference-based today: `coeus_autograd::Var::grad()`
// returns a plain non-differentiable `Tensor`, so there is no double-backward
// and `autodiff_second_derivative_xx` reconstructs the second derivative by
// differencing (see `ml::autodiff_utils::second_order`). When coeus_autograd
// gains double-backward this becomes a genuine autodiff-vs-FD comparison
// rather than two spellings of the same method, and the tolerance should be
// re-derived at that point.
#[test]
fn test_second_derivative_xx_vs_finite_difference() {
    let config = Config::default();

    let model_autodiff = ElasticPINN2D::<B>::new(&config).unwrap();
    let model_fd = ElasticPINN2D::<B>::new(&config).unwrap();

    // Deliberately not extended with the deleted analytic test's points.
    // `REL_TOL_SECOND` is empirical, not derived: both sides here are finite
    // differences of the same function at different step sizes, so their
    // disagreement scales with the fourth derivative at the point, which
    // nothing bounds. Adding (0.3, 0.5, 0.1) measured rel_err 2.59e-2 against
    // the 1e-2 bound -- not a defect in the method, and not a reason to widen
    // a tolerance to fit it either. Tracked as KW-LINT-047's finding.
    let test_points = vec![(0.5, 0.5, 0.5), (0.3, 0.7, 0.2)];

    for (x, y, t) in test_points {
        for component in 0..2 {
            let autodiff_second =
                autodiff_second_derivative_xx(&model_autodiff, x, y, t, component).unwrap();
            let fd_second = second_difference_xx(&model_fd, x, y, t, component, FD_H_SECOND);

            let abs_error = (autodiff_second - fd_second).abs();
            let rel_error = abs_error / (fd_second.abs() + 1e-8);

            assert!(
                rel_error < REL_TOL_SECOND || abs_error < 1e-5,
                "Second derivative mismatch: autodiff={:.6e}, FD={:.6e}, rel_err={:.6e}",
                autodiff_second,
                fd_second,
                rel_error
            );
        }
    }
}
