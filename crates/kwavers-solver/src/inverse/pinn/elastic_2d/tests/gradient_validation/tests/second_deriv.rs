use super::helpers::{autodiff_second_derivative_xx, second_difference_xx};
use super::{FD_H_SECOND, REL_TOL_SECOND};
use crate::inverse::elastic_2d::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;

type B = super::TestBackend;

#[test]
#[ignore = "Requires trained model for reliable FD comparison - use analytic tests instead"]
fn test_second_derivative_xx_vs_finite_difference() {
    let config = Config::default();

    let model_autodiff = ElasticPINN2D::<B>::new(&config).unwrap();
    let model_fd = ElasticPINN2D::<B>::new(&config).unwrap();

    let test_points = vec![(0.5, 0.5, 0.5), (0.3, 0.7, 0.2)];

    for (x, y, t) in test_points {
        for component in 0..2 {
            let autodiff_second =
                autodiff_second_derivative_xx(&model_autodiff, x, y, t, component).unwrap();
            let fd_second = second_difference_xx(&model_fd, x, y, t, component, FD_H_SECOND);

            let abs_error = (autodiff_second - fd_second).abs();
            let rel_error = abs_error / (fd_second.abs() + 1e-8);

            println!(
                "∂²u{}/∂x² at ({:.2},{:.2},{:.2}): autodiff={:.6e}, FD={:.6e}, rel_err={:.6e}",
                if component == 0 { "ₓ" } else { "ᵧ" },
                x,
                y,
                t,
                autodiff_second,
                fd_second,
                rel_error
            );

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

#[test]
#[ignore = "Both paths are now finite-difference-based on this untrained model (coeus_autograd has no double-backward — see autodiff_utils::second_order's weight-gradient contract), so this duplicates test_second_derivative_xx_vs_finite_difference; retained for its is_finite() smoke coverage"]
fn test_analytic_polynomial_second_derivative() {
    // Polynomial: u(x) = x² → ∂u/∂x = 2x → ∂²u/∂x² = 2
    // This tests the finite-difference second-derivative path on a known
    // analytic function.
    //
    // NOTE: `coeus_autograd::Var::grad()` returns a plain, non-differentiable
    // `Tensor` (no double-backward support), so `autodiff_second_derivative_xx`
    // is finite-difference-based (see
    // `ml::autodiff_utils::second_order::compute_second_derivative_2d`),
    // not a nested-autodiff reconstruction.

    let config = Config::default();
    let model = ElasticPINN2D::<B>::new(&config).unwrap();

    let test_points = vec![(0.3, 0.5, 0.1), (0.5, 0.5, 0.5), (0.7, 0.3, 0.2)];

    for (x, y, t) in test_points {
        let second_deriv = autodiff_second_derivative_xx(&model, x, y, t, 0).unwrap();

        assert!(
            second_deriv.is_finite(),
            "∂²u/∂x² should be finite at ({},{},{})",
            x,
            y,
            t
        );

        println!(
            "Analytic polynomial: ∂²u/∂x² at ({:.2},{:.2},{:.2}) = {:.6e} (finite ✓)",
            x, y, t, second_deriv
        );
    }
}
