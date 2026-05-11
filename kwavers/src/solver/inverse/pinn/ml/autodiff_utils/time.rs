//! Time-derivative autodiff utilities: ∂u/∂t and ∂²u/∂t².
//!
//! # References
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.

use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute first-order time derivative ∂u/∂t.
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model.
/// - `input`: Input tensor of shape `[batch, 3]` where columns are `[t, x, y]`.
/// - `output_component`: Which output component to differentiate (0 for u_x, 1 for u_y).
///
/// # Returns
/// Tensor of shape `[batch, 1]` containing ∂u/∂t for the specified component.
///
/// # Burn 0.19+ autodiff pattern
/// ```text
/// input_grad = input.require_grad()
/// output = forward(input_grad)
/// grads = output.sum().backward()
/// du_dt = input_grad.grad(&grads)[:, 0]   // column 0 = time
/// ```
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn compute_time_derivative<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
) -> Result<Tensor<B::InnerBackend, 2>, crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    let input_grad = input.clone().require_grad();
    let output = forward_fn(input_grad.clone());
    let component = output
        .clone()
        .slice([0..output.dims()[0], output_component..output_component + 1]);
    let grads = component.sum().backward();
    let dt_grad = input_grad
        .grad(&grads)
        .ok_or_else(|| {
            crate::error::KwaversError::InternalError(
                "Failed to compute time derivative gradient".into(),
            )
        })?
        .slice([0..input.dims()[0], 0..1]);
    Ok(dt_grad)
}

/// Compute second-order time derivative ∂²u/∂t² via central finite differences.
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model.
/// - `input`: Input tensor `[batch, 3]`.
/// - `output_component`: Output component index.
///
/// # Returns
/// Tensor `[batch, 1]` containing ∂²u/∂t².
///
/// # Mathematical Note
/// Central finite-difference approximation (ε = 1e-4):
/// ```text
/// ∂²u/∂t² ≈ (u(t+ε) − 2u(t) + u(t−ε)) / ε²
/// ```
/// Truncation error O(ε²). Nested autodiff in Burn requires InnerBackend tensor handling;
/// finite differences provide a pragmatic alternative on non-throughput-critical PDE paths.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn compute_second_time_derivative<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
) -> Result<Tensor<B::InnerBackend, 2>, crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2> + Clone,
{
    let eps = 1e-4;
    let batch = input.dims()[0];

    let t_col = input.clone().slice([0..batch, 0..1]);
    let x_col = input.clone().slice([0..batch, 1..2]);
    let y_col = input.clone().slice([0..batch, 2..3]);

    let t_plus = t_col.clone().add_scalar(eps);
    let t_minus = t_col.sub_scalar(eps);

    let input_plus = Tensor::cat(vec![t_plus, x_col.clone(), y_col.clone()], 1);
    let input_minus = Tensor::cat(vec![t_minus, x_col, y_col], 1);

    let output = forward_fn(input.clone());
    let u_t = output
        .clone()
        .slice([0..output.dims()[0], output_component..output_component + 1]);

    let output_plus = forward_fn(input_plus);
    let u_t_plus = output_plus.clone().slice([
        0..output_plus.dims()[0],
        output_component..output_component + 1,
    ]);

    let output_minus = forward_fn(input_minus);
    let u_t_minus = output_minus.clone().slice([
        0..output_minus.dims()[0],
        output_component..output_component + 1,
    ]);

    let d2u_dt2 = (u_t_plus + u_t_minus - u_t * 2.0) / (eps * eps);
    Ok(d2u_dt2.inner())
}
