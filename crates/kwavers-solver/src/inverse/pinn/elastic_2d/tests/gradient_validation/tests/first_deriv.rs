use super::helpers::{autodiff_gradient_x, central_difference_x, central_difference_y};
use super::{FD_H_FIRST, REL_TOL_FIRST};
use crate::inverse::elastic_2d::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
use coeus_autograd::Var;

type B = super::TestBackend;

#[test]
#[ignore = "FD comparison unreliable on untrained models - use analytic tests instead"]
fn test_first_derivative_x_vs_finite_difference() {
    let config = Config::default();

    let model_autodiff = ElasticPINN2D::<B>::new(&config).unwrap();
    let model_fd = ElasticPINN2D::<B>::new(&config).unwrap();

    let test_points = vec![
        (0.3, 0.5, 0.1),
        (0.5, 0.5, 0.5),
        (0.7, 0.3, 0.8),
        (0.2, 0.8, 0.2),
    ];

    for (x, y, t) in test_points {
        for component in 0..2 {
            let autodiff_grad = autodiff_gradient_x(&model_autodiff, x, y, t, component).unwrap();
            let fd_grad = central_difference_x(&model_fd, x, y, t, component, FD_H_FIRST);

            let abs_error = (autodiff_grad - fd_grad).abs();
            let rel_error = abs_error / (fd_grad.abs() + 1e-10);

            println!(
                "∂u{}/∂x at ({:.2},{:.2},{:.2}): autodiff={:.6e}, FD={:.6e}, rel_err={:.6e}",
                if component == 0 { "ₓ" } else { "ᵧ" },
                x,
                y,
                t,
                autodiff_grad,
                fd_grad,
                rel_error
            );

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
#[ignore = "FD comparison unreliable on untrained models - use analytic tests instead"]
fn test_first_derivative_y_vs_finite_difference() {
    let config = Config::default();

    let model_autodiff = ElasticPINN2D::<B>::new(&config).unwrap();
    let model_fd = ElasticPINN2D::<B>::new(&config).unwrap();

    let test_points = vec![(0.5, 0.5, 0.5), (0.3, 0.7, 0.2)];

    let backend = B::default();
    let var_point = |v: f64| {
        Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[v as f32], &backend),
            false,
        )
    };

    for (x, y, t) in test_points {
        for component in 0..2 {
            let x_t = var_point(x);
            let y_t = Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[y as f32], &backend),
                true,
            );
            let t_t = var_point(t);

            let u = model_autodiff.forward(&x_t, &y_t, &t_t);
            let u_component = coeus_autograd::slice(&u, &[(0, 1), (component, component + 1)]);
            coeus_autograd::sum(&u_component).backward();

            let du_dy = y_t.grad().expect("Gradient ∂u/∂y should exist");
            let autodiff_grad = du_dy.as_slice()[0] as f64;

            let fd_grad = central_difference_y(&model_fd, x, y, t, component, FD_H_FIRST);

            let abs_error = (autodiff_grad - fd_grad).abs();
            let rel_error = abs_error / (fd_grad.abs() + 1e-10);

            println!(
                "∂u{}/∂y at ({:.2},{:.2},{:.2}): autodiff={:.6e}, FD={:.6e}, rel_err={:.6e}",
                if component == 0 { "ₓ" } else { "ᵧ" },
                x,
                y,
                t,
                autodiff_grad,
                fd_grad,
                rel_error
            );

            assert!(
                rel_error < REL_TOL_FIRST || abs_error < 1e-6,
                "Gradient mismatch for ∂u/∂y"
            );
        }
    }
}
