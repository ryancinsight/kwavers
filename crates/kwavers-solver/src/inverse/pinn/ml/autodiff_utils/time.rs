//! Time-derivative autodiff utilities: ∂u/∂t and ∂²u/∂t².
//!
//! # References
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.

use coeus_autograd::Var;

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
/// # coeus_autograd pattern
/// ```text
/// input_grad = Var::new(input.clone(), true)
/// output = forward(&input_grad)
/// sum(output_component).backward()
/// du_dt = input_grad.grad()[:, 0]   // column 0 = time
/// ```
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_time_derivative<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    output_component: usize,
) -> Result<coeus_tensor::Tensor<f32, B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B>,
{
    let batch = input.shape()[0];
    let input_grad = Var::new(input.clone(), true);
    let output = forward_fn(&input_grad);
    let component = coeus_autograd::slice(&output, &[(0, batch), (output_component, output_component + 1)]);
    coeus_autograd::sum(&component).backward();
    let dt_grad = input_grad
        .grad()
        .ok_or_else(|| {
            kwavers_core::error::KwaversError::InternalError(
                "Failed to compute time derivative gradient".into(),
            )
        })?
        .slice(&[(0, batch), (0, 1)])
        .to_contiguous();
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
/// Truncation error O(ε²).
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
pub fn compute_second_time_derivative<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    output_component: usize,
) -> Result<coeus_tensor::Tensor<f32, B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B>,
{
    let eps = 1e-4_f32;
    let batch = input.shape()[0];
    let backend = B::default();

    let raw = input.as_slice();
    let mut plus = raw.to_vec();
    let mut minus = raw.to_vec();
    for row in 0..batch {
        plus[row * 3] += eps;
        minus[row * 3] -= eps;
    }
    let input_plus = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &plus, &backend);
    let input_minus = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &minus, &backend);

    let output = forward_fn(&Var::new(input.clone(), false));
    let u_t = coeus_autograd::slice(&output, &[(0, batch), (output_component, output_component + 1)]);

    let output_plus = forward_fn(&Var::new(input_plus, false));
    let u_t_plus = coeus_autograd::slice(
        &output_plus,
        &[(0, batch), (output_component, output_component + 1)],
    );

    let output_minus = forward_fn(&Var::new(input_minus, false));
    let u_t_minus = coeus_autograd::slice(
        &output_minus,
        &[(0, batch), (output_component, output_component + 1)],
    );

    let two_u = coeus_autograd::scalar_mul(&u_t, 2.0);
    let d2u_dt2 = coeus_autograd::scalar_mul(
        &coeus_autograd::sub(&coeus_autograd::add(&u_t_plus, &u_t_minus), &two_u),
        1.0 / (eps * eps),
    );
    Ok(d2u_dt2.tensor)
}
