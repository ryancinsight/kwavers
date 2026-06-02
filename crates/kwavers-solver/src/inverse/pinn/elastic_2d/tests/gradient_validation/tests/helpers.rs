use super::{TestAutodiffBackend, TestBackend};
use kwavers_core::error::KwaversResult;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
use burn::tensor::Tensor;

/// Compute central finite difference approximation of ∂f/∂x
///
/// # Mathematical Formula
///
/// ```text
/// ∂f/∂x ≈ (f(x+h) - f(x-h)) / (2h)
/// ```
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub(super) fn central_difference_x(
    model: &ElasticPINN2D<TestBackend>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
    h: f64,
) -> f64 {
    let device = Default::default();

    let x_plus = Tensor::<TestBackend, 2>::from_floats([[(x + h) as f32]], &device);
    let y_t = Tensor::<TestBackend, 2>::from_floats([[y as f32]], &device);
    let t_t = Tensor::<TestBackend, 2>::from_floats([[t as f32]], &device);
    let u_plus = model.forward(x_plus, y_t.clone(), t_t.clone());

    let x_minus = Tensor::<TestBackend, 2>::from_floats([[(x - h) as f32]], &device);
    let u_minus = model.forward(x_minus, y_t, t_t);

    let u_plus_val = u_plus.to_data().as_slice::<f32>().unwrap()[component] as f64;
    let u_minus_val = u_minus.to_data().as_slice::<f32>().unwrap()[component] as f64;

    (u_plus_val - u_minus_val) / (2.0 * h)
}

/// Compute central finite difference approximation of ∂f/∂y
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub(super) fn central_difference_y(
    model: &ElasticPINN2D<TestBackend>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
    h: f64,
) -> f64 {
    let device = Default::default();

    let x_t = Tensor::<TestBackend, 2>::from_floats([[x as f32]], &device);
    let y_plus = Tensor::<TestBackend, 2>::from_floats([[(y + h) as f32]], &device);
    let t_t = Tensor::<TestBackend, 2>::from_floats([[t as f32]], &device);
    let u_plus = model.forward(x_t.clone(), y_plus, t_t.clone());

    let y_minus = Tensor::<TestBackend, 2>::from_floats([[(y - h) as f32]], &device);
    let u_minus = model.forward(x_t, y_minus, t_t);

    let u_plus_val = u_plus.to_data().as_slice::<f32>().unwrap()[component] as f64;
    let u_minus_val = u_minus.to_data().as_slice::<f32>().unwrap()[component] as f64;

    (u_plus_val - u_minus_val) / (2.0 * h)
}

/// Compute second derivative ∂²f/∂x² using finite differences
///
/// # Mathematical Formula
///
/// ```text
/// ∂²f/∂x² ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
/// ```
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub(super) fn second_difference_xx(
    model: &ElasticPINN2D<TestBackend>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
    h: f64,
) -> f64 {
    let device = Default::default();

    let y_t = Tensor::<TestBackend, 2>::from_floats([[y as f32]], &device);
    let t_t = Tensor::<TestBackend, 2>::from_floats([[t as f32]], &device);

    let x_plus = Tensor::<TestBackend, 2>::from_floats([[(x + h) as f32]], &device);
    let u_plus = model.forward(x_plus, y_t.clone(), t_t.clone());

    let x_center = Tensor::<TestBackend, 2>::from_floats([[x as f32]], &device);
    let u_center = model.forward(x_center, y_t.clone(), t_t.clone());

    let x_minus = Tensor::<TestBackend, 2>::from_floats([[(x - h) as f32]], &device);
    let u_minus = model.forward(x_minus, y_t, t_t);

    let u_plus_val = u_plus.to_data().as_slice::<f32>().unwrap()[component] as f64;
    let u_center_val = u_center.to_data().as_slice::<f32>().unwrap()[component] as f64;
    let u_minus_val = u_minus.to_data().as_slice::<f32>().unwrap()[component] as f64;

    (u_plus_val - 2.0 * u_center_val + u_minus_val) / (h * h)
}

/// Compute autodiff gradient ∂u/∂x at a point
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # Panics
/// - Panics if `Gradient should exist`.
///
pub(super) fn autodiff_gradient_x(
    model: &ElasticPINN2D<TestAutodiffBackend>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
) -> KwaversResult<f64> {
    let device = Default::default();

    let x_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[x as f32]], &device).require_grad();
    let y_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[y as f32]], &device);
    let t_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[t as f32]], &device);

    let u = model.forward(x_t.clone(), y_t, t_t);
    let u_component = u.slice([0..1, component..component + 1]);

    let grads = u_component.backward();
    let du_dx_inner = x_t.grad(&grads).expect("Gradient should exist");

    let du_dx =
        Tensor::<TestAutodiffBackend, 2>::from_data(du_dx_inner.into_data(), &Default::default());
    let du_dx_val = du_dx.to_data().as_slice::<f32>().unwrap()[0] as f64;

    Ok(du_dx_val)
}

/// Compute autodiff gradient ∂u/∂y at a point
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # Panics
/// - Panics if `Gradient should exist`.
///
pub(super) fn autodiff_gradient_y(
    model: &ElasticPINN2D<TestAutodiffBackend>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
) -> KwaversResult<f64> {
    let device = Default::default();

    let x_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[x as f32]], &device);
    let y_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[y as f32]], &device).require_grad();
    let t_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[t as f32]], &device);

    let u = model.forward(x_t, y_t.clone(), t_t);
    let u_component = u.slice([0..1, component..component + 1]);

    let grads = u_component.backward();
    let du_dy_inner = y_t.grad(&grads).expect("Gradient should exist");

    let du_dy =
        Tensor::<TestAutodiffBackend, 2>::from_data(du_dy_inner.into_data(), &Default::default());
    let du_dy_val = du_dy.to_data().as_slice::<f32>().unwrap()[0] as f64;

    Ok(du_dy_val)
}

/// Compute autodiff second derivative ∂²u/∂x²
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # Panics
/// - Panics if `First gradient should exist`.
/// - Panics if `Second gradient should exist`.
///
pub(super) fn autodiff_second_derivative_xx(
    model: &ElasticPINN2D<TestAutodiffBackend>,
    x: f64,
    y: f64,
    t: f64,
    component: usize,
) -> KwaversResult<f64> {
    let device = Default::default();

    let x_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[x as f32]], &device).require_grad();
    let y_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[y as f32]], &device);
    let t_t = Tensor::<TestAutodiffBackend, 2>::from_floats([[t as f32]], &device);

    let u = model.forward(x_t.clone(), y_t, t_t);
    let u_component = u.slice([0..1, component..component + 1]);

    let grads_first = u_component.backward();
    let du_dx_inner = x_t.grad(&grads_first).expect("First gradient should exist");
    let du_dx =
        Tensor::<TestAutodiffBackend, 2>::from_data(du_dx_inner.into_data(), &Default::default())
            .require_grad();

    let grads_second = du_dx.backward();
    let d2u_dx2_inner = x_t
        .grad(&grads_second)
        .expect("Second gradient should exist");
    let d2u_dx2 =
        Tensor::<TestAutodiffBackend, 2>::from_data(d2u_dx2_inner.into_data(), &Default::default());

    let d2u_dx2_val = d2u_dx2.to_data().as_slice::<f32>().unwrap()[0] as f64;

    Ok(d2u_dx2_val)
}
