use super::helpers::{
    autodiff_gradient_x, autodiff_gradient_y, central_difference_x, central_difference_y,
};
use super::{FD_H_FIRST, REL_TOL_FIRST};
use crate::inverse::elastic_2d::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;

type B = super::TestBackend;

// Autodiff-vs-central-difference gradient check on identically initialized
// models. Measured 2026-08-22: ~20 ms, stable across 5/5 runs (rel tol 1e-3,
// h = 1e-5), so the previous ignore is lifted.
#[test]
fn test_first_derivative_x_vs_finite_difference() {
    let config = Config::default();

    let model_autodiff = ElasticPINN2D::<B>::new(&config).unwrap();
    let model_fd = ElasticPINN2D::<B>::new(&config).unwrap();

    // The last two came from `analytic.rs`, which claimed to check these
    // points against a closed-form sine derivative but asserted only
    // finiteness. The points are worth keeping; the oracle they now run
    // against is the one that can actually fail.
    let test_points = vec![
        (0.3, 0.5, 0.1),
        (0.5, 0.5, 0.5),
        (0.7, 0.3, 0.8),
        (0.2, 0.8, 0.2),
        (0.0, 0.5, 0.5),
        (0.25, 0.5, 0.5),
    ];

    for (x, y, t) in test_points {
        for component in 0..2 {
            let autodiff_grad = autodiff_gradient_x(&model_autodiff, x, y, t, component).unwrap();
            let fd_grad = central_difference_x(&model_fd, x, y, t, component, FD_H_FIRST);

            let abs_error = (autodiff_grad - fd_grad).abs();
            let rel_error = abs_error / (fd_grad.abs() + 1e-10);

            assert!(
                rel_error < REL_TOL_FIRST || abs_error < 1e-6,
                "Gradient mismatch: autodiff={:.6e}, FD={:.6e}, rel_err={:.6e} at ({},{},{})",
                autodiff_grad,
                fd_grad,
                rel_error,
                x,
                y,
                t
            );
        }
    }
}

#[test]
fn test_first_derivative_y_vs_finite_difference() {
    let config = Config::default();

    let model_autodiff = ElasticPINN2D::<B>::new(&config).unwrap();
    let model_fd = ElasticPINN2D::<B>::new(&config).unwrap();

    let test_points = vec![(0.5, 0.5, 0.5), (0.3, 0.7, 0.2)];

    for (x, y, t) in test_points {
        for component in 0..2 {
            let autodiff_grad = autodiff_gradient_y(&model_autodiff, x, y, t, component).unwrap();
            let fd_grad = central_difference_y(&model_fd, x, y, t, component, FD_H_FIRST);

            let abs_error = (autodiff_grad - fd_grad).abs();
            let rel_error = abs_error / (fd_grad.abs() + 1e-10);

            assert!(
                rel_error < REL_TOL_FIRST || abs_error < 1e-6,
                "∂u/∂y mismatch: autodiff={autodiff_grad:.6e}, FD={fd_grad:.6e},                  rel_err={rel_error:.6e} at ({x},{y},{t}), component {component}"
            );
        }
    }
}
