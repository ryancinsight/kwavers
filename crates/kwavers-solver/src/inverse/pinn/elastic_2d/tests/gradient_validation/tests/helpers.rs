use super::TestBackend;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
use crate::inverse::pinn::ml::autodiff_utils::compute_second_derivative_2d;
use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

type B = TestBackend;

fn var_point(v: f64) -> Var<f32, B> {
    let backend = B::default();
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[v as f32], &backend),
        false,
    )
}

/// Compute central finite difference approximation of ∂f/∂x
///
/// # Mathematical Formula
///
/// ```text
/// ∂f/∂x ≈ (f(x+h) - f(x-h)) / (2h)
/// ```
pub(super) fn central_difference_x(
    model: &ElasticPINN2D<B>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
    h: f64,
) -> f64 {
    let y_t = var_point(y);
    let t_t = var_point(t);
    let u_plus = model.forward(&var_point(x + h), &y_t, &t_t);
    let u_minus = model.forward(&var_point(x - h), &y_t, &t_t);

    let u_plus_val = u_plus.tensor.as_slice()[component] as f64;
    let u_minus_val = u_minus.tensor.as_slice()[component] as f64;

    (u_plus_val - u_minus_val) / (2.0 * h)
}

/// Compute central finite difference approximation of ∂f/∂y
pub(super) fn central_difference_y(
    model: &ElasticPINN2D<B>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
    h: f64,
) -> f64 {
    let x_t = var_point(x);
    let t_t = var_point(t);
    let u_plus = model.forward(&x_t, &var_point(y + h), &t_t);
    let u_minus = model.forward(&x_t, &var_point(y - h), &t_t);

    let u_plus_val = u_plus.tensor.as_slice()[component] as f64;
    let u_minus_val = u_minus.tensor.as_slice()[component] as f64;

    (u_plus_val - u_minus_val) / (2.0 * h)
}

/// Compute second derivative ∂²f/∂x² using finite differences
///
/// # Mathematical Formula
///
/// ```text
/// ∂²f/∂x² ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
/// ```
pub(super) fn second_difference_xx(
    model: &ElasticPINN2D<B>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
    h: f64,
) -> f64 {
    let y_t = var_point(y);
    let t_t = var_point(t);

    let u_plus = model.forward(&var_point(x + h), &y_t, &t_t);
    let u_center = model.forward(&var_point(x), &y_t, &t_t);
    let u_minus = model.forward(&var_point(x - h), &y_t, &t_t);

    let u_plus_val = u_plus.tensor.as_slice()[component] as f64;
    let u_center_val = u_center.tensor.as_slice()[component] as f64;
    let u_minus_val = u_minus.tensor.as_slice()[component] as f64;

    (u_plus_val - 2.0 * u_center_val + u_minus_val) / (h * h)
}

/// Compute autodiff gradient ∂u/∂x at a point via true (first-order) reverse-mode autodiff.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
pub(super) fn autodiff_gradient_x(
    model: &ElasticPINN2D<B>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
) -> KwaversResult<f64> {
    let x_t = Var::new(x_leaf(x), true);
    let y_t = var_point(y);
    let t_t = var_point(t);

    let u = model.forward(&x_t, &y_t, &t_t);
    let u_component = coeus_autograd::slice(&u, &[(0, 1), (component, component + 1)]);
    coeus_autograd::sum(&u_component).backward();

    let du_dx = x_t.grad().expect("Gradient should exist");
    Ok(du_dx.as_slice()[0] as f64)
}

/// Compute autodiff gradient ∂u/∂y at a point via true (first-order) reverse-mode autodiff.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
pub(super) fn autodiff_gradient_y(
    model: &ElasticPINN2D<B>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
) -> KwaversResult<f64> {
    let x_t = var_point(x);
    let y_t = Var::new(x_leaf(y), true);
    let t_t = var_point(t);

    let u = model.forward(&x_t, &y_t, &t_t);
    let u_component = coeus_autograd::slice(&u, &[(0, 1), (component, component + 1)]);
    coeus_autograd::sum(&u_component).backward();

    let du_dy = y_t.grad().expect("Gradient should exist");
    Ok(du_dy.as_slice()[0] as f64)
}

fn x_leaf(v: f64) -> coeus_tensor::Tensor<f32, B> {
    let backend = B::default();
    coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[v as f32], &backend)
}

/// Compute second derivative ∂²u/∂x² at a point.
///
/// `coeus_autograd` has no double-backward support (`Var::grad()` returns a
/// plain, non-differentiable `Tensor` — see
/// [`crate::inverse::pinn::ml::autodiff_utils::second_order`]'s
/// module-level weight-gradient contract), so this delegates to
/// [`compute_second_derivative_2d`]'s finite-difference implementation
/// rather than a nested-autodiff reconstruction.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
pub(super) fn autodiff_second_derivative_xx(
    model: &ElasticPINN2D<B>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
) -> KwaversResult<f64> {
    let backend = B::default();
    let input = coeus_tensor::Tensor::from_slice_on(vec![1, 3], &[t as f32, x as f32, y as f32], &backend);
    let forward = |combined: &Var<f32, B>| -> Var<f32, B> {
        let n = combined.tensor.shape()[0];
        let t = coeus_autograd::slice(combined, &[(0, n), (0, 1)]);
        let x = coeus_autograd::slice(combined, &[(0, n), (1, 2)]);
        let y = coeus_autograd::slice(combined, &[(0, n), (2, 3)]);
        model.forward(&x, &y, &t)
    };
    let d2u_dx2 = compute_second_derivative_2d(forward, &input, component, 1)?;
    Ok(d2u_dx2.tensor.as_slice()[0] as f64)
}
