//! First-order spatial autodiff utilities: ∂u/∂x, ∂u/∂y, and ∇·u.
//!
//! # References
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.

use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute spatial gradient ∂u/∂x and ∂u/∂y for a 2D displacement field.
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]` with columns `[t, x, y]`.
/// - `output_component`: Output component (0 for u_x, 1 for u_y).
///
/// # Returns
/// Tuple `(∂u/∂x, ∂u/∂y)`, each of shape `[batch, 1]`.
///
/// # Burn 0.19+ autodiff pattern
/// Columns 1 and 2 of `input.grad(&grads)` yield ∂/∂x and ∂/∂y respectively.
pub fn compute_spatial_gradient_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
) -> Result<(Tensor<B::InnerBackend, 2>, Tensor<B::InnerBackend, 2>), crate::error::KwaversError>
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

    let grad_tensor = input_grad.grad(&grads).ok_or_else(|| {
        crate::error::KwaversError::InternalError("Failed to compute spatial gradient".into())
    })?;

    let dx_grad = grad_tensor.clone().slice([0..input.dims()[0], 1..2]);
    let dy_grad = grad_tensor.slice([0..input.dims()[0], 2..3]);

    Ok((dx_grad, dy_grad))
}

/// Compute divergence ∇·u = ∂u_x/∂x + ∂u_y/∂y.
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]`.
///
/// # Returns
/// Tensor `[batch, 1]` containing the divergence field.
///
/// # Mathematical Note
/// ```text
/// ∇·u = ∂u_x/∂x + ∂u_y/∂y
/// ```
/// Two separate backward passes are required because each component's gradient
/// is computed from a distinct scalar reduction of its corresponding output slice.
pub fn compute_divergence_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
) -> Result<Tensor<B::InnerBackend, 2>, crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    // ∂u_x/∂x — backward through u_x component
    let input_grad = input.clone().require_grad();
    let output = forward_fn(input_grad.clone());
    let u_x = output.clone().slice([0..output.dims()[0], 0..1]);
    let grads_x = u_x.sum().backward();
    let du_x_dx = input_grad
        .grad(&grads_x)
        .ok_or_else(|| {
            crate::error::KwaversError::InternalError("Failed to compute ∂u_x/∂x gradient".into())
        })?
        .slice([0..input.dims()[0], 1..2]);

    // ∂u_y/∂y — backward through u_y component
    let input_grad = input.clone().require_grad();
    let output = forward_fn(input_grad.clone());
    let u_y = output.clone().slice([0..output.dims()[0], 1..2]);
    let grads_y = u_y.sum().backward();
    let du_y_dy = input_grad
        .grad(&grads_y)
        .ok_or_else(|| {
            crate::error::KwaversError::InternalError("Failed to compute ∂u_y/∂y gradient".into())
        })?
        .slice([0..input.dims()[0], 2..3]);

    Ok(du_x_dx + du_y_dy)
}
