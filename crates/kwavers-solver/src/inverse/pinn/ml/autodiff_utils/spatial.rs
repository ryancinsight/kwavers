//! First-order spatial autodiff utilities: ∂u/∂x, ∂u/∂y, and ∇·u.
//!
//! # References
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.

use coeus_autograd::Var;

/// 2D gradient component pair `(∂u/∂x, ∂u/∂y)`, each `[batch, 1]`.
type GradientPair2D<B> = (coeus_tensor::Tensor<f32, B>, coeus_tensor::Tensor<f32, B>);

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
/// # coeus_autograd pattern
/// A fresh leaf `Var` tracks the input; after `.backward()` on the summed
/// output component, columns 1 and 2 of `input_grad.grad()` yield ∂/∂x and
/// ∂/∂y respectively.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_spatial_gradient_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    output_component: usize,
) -> Result<GradientPair2D<B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B>,
{
    let batch = input.shape()[0];
    let input_grad = Var::new(input.clone(), true);
    let output = forward_fn(&input_grad);
    let component = coeus_autograd::slice(
        &output,
        &[(0, batch), (output_component, output_component + 1)],
    );
    coeus_autograd::sum(&component).backward();

    let grad_tensor = input_grad.grad().ok_or_else(|| {
        kwavers_core::error::KwaversError::InternalError(
            "Failed to compute spatial gradient".into(),
        )
    })?;

    let dx_grad = grad_tensor.slice(&[(0, batch), (1, 2)]).unwrap().to_contiguous();
    let dy_grad = grad_tensor.slice(&[(0, batch), (2, 3)]).unwrap().to_contiguous();

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
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_divergence_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
) -> Result<coeus_tensor::Tensor<f32, B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B>,
{
    let batch = input.shape()[0];

    // ∂u_x/∂x — backward through u_x component
    let input_grad = Var::new(input.clone(), true);
    let output = forward_fn(&input_grad);
    let u_x = coeus_autograd::slice(&output, &[(0, batch), (0, 1)]);
    coeus_autograd::sum(&u_x).backward();
    let du_x_dx = input_grad
        .grad()
        .ok_or_else(|| {
            kwavers_core::error::KwaversError::InternalError(
                "Failed to compute ∂u_x/∂x gradient".into(),
            )
        })?
        .slice(&[(0, batch), (1, 2)])
        .to_contiguous();

    // ∂u_y/∂y — backward through u_y component
    let input_grad = Var::new(input.clone(), true);
    let output = forward_fn(&input_grad);
    let u_y = coeus_autograd::slice(&output, &[(0, batch), (1, 2)]);
    coeus_autograd::sum(&u_y).backward();
    let du_y_dy = input_grad
        .grad()
        .ok_or_else(|| {
            kwavers_core::error::KwaversError::InternalError(
                "Failed to compute ∂u_y/∂y gradient".into(),
            )
        })?
        .slice(&[(0, batch), (2, 3)])
        .to_contiguous();

    let backend = B::default();
    Ok(coeus_ops::add(&du_x_dx, &du_y_dy, &backend))
}
