//! Second-order spatial autodiff utilities: ∂²u/∂xᵢ², ∇²u, and ∇(∇·u).
//!
//! # References
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.

use coeus_autograd::Var;

use super::spatial::compute_divergence_2d;

/// 2D gradient component pair `(∂/∂x, ∂/∂y)`, each `[batch, 1]`.
type GradientPair2D<B> = (coeus_tensor::Tensor<f32, B>, coeus_tensor::Tensor<f32, B>);

/// Compute second-order spatial derivative ∂²u/∂xᵢ² via central finite differences.
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]`.
/// - `output_component`: Output component (0 for u_x, 1 for u_y).
/// - `spatial_dim`: Spatial dimension to differentiate twice (1 for x, 2 for y).
///
/// # Returns
/// Tensor `[batch, 1]` containing the second derivative.
///
/// # Mathematical Note
/// Central finite-difference approximation (ε = 1e-4):
/// ```text
/// ∂²u/∂xᵢ² ≈ (u(xᵢ+ε) − 2u(xᵢ) + u(xᵢ−ε)) / ε²
/// ```
/// Truncation error O(ε²).
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
pub fn compute_second_derivative_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    output_component: usize,
    spatial_dim: usize,
) -> Result<coeus_tensor::Tensor<f32, B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B>,
{
    if !(1..=2).contains(&spatial_dim) {
        return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
            "spatial_dim must be 1 (x) or 2 (y), got {}",
            spatial_dim
        )));
    }

    let eps = 1e-4_f32;
    let batch = input.shape()[0];
    let backend = B::default();

    let raw = input.as_slice();
    let mut plus = raw.to_vec();
    let mut minus = raw.to_vec();
    for row in 0..batch {
        plus[row * 3 + spatial_dim] += eps;
        minus[row * 3 + spatial_dim] -= eps;
    }
    let input_plus = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &plus, &backend);
    let input_minus = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &minus, &backend);

    let output = forward_fn(&Var::new(input.clone(), false));
    let u = coeus_autograd::slice(&output, &[(0, batch), (output_component, output_component + 1)]);

    let output_plus = forward_fn(&Var::new(input_plus, false));
    let u_plus = coeus_autograd::slice(
        &output_plus,
        &[(0, batch), (output_component, output_component + 1)],
    );

    let output_minus = forward_fn(&Var::new(input_minus, false));
    let u_minus = coeus_autograd::slice(
        &output_minus,
        &[(0, batch), (output_component, output_component + 1)],
    );

    let two_u = coeus_autograd::scalar_mul(&u, 2.0);
    let d2u = coeus_autograd::scalar_mul(
        &coeus_autograd::sub(&coeus_autograd::add(&u_plus, &u_minus), &two_u),
        1.0 / (eps * eps),
    );
    Ok(d2u.tensor)
}

/// Compute scalar Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y².
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]`.
/// - `output_component`: Output component.
///
/// # Returns
/// Tensor `[batch, 1]` containing the Laplacian.
///
/// # Mathematical Note
/// ```text
/// ∇²u = ∂²u/∂x² + ∂²u/∂y²
/// ```
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_laplacian_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    output_component: usize,
) -> Result<coeus_tensor::Tensor<f32, B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B> + Clone,
{
    let backend = B::default();
    let d2u_dx2 = compute_second_derivative_2d(forward_fn.clone(), input, output_component, 1)?;
    let d2u_dy2 = compute_second_derivative_2d(forward_fn, input, output_component, 2)?;
    Ok(coeus_ops::add(&d2u_dx2, &d2u_dy2, &backend))
}

/// Compute gradient of divergence ∇(∇·u) = [∂(∇·u)/∂x, ∂(∇·u)/∂y] for the P-wave term.
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]`.
///
/// # Returns
/// Tuple `(∂(∇·u)/∂x, ∂(∇·u)/∂y)`.
///
/// # Mathematical Specification
/// Elastic wave P-wave term (Achenbach 1973):
/// ```text
/// (λ + 2μ)∇(∇·u) = (λ + 2μ)[∂(∇·u)/∂x, ∂(∇·u)/∂y]
/// ```
/// Each partial derivative is approximated via forward finite difference (ε = 1e-5)
/// applied to `compute_divergence_2d`.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_gradient_of_divergence_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
) -> Result<GradientPair2D<B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B> + Clone,
{
    let eps = 1e-5_f32;
    let batch = input.shape()[0];
    let backend = B::default();

    let div_center = compute_divergence_2d(forward_fn.clone(), input)?;
    let div_center_var = Var::new(div_center, false);

    // ∂(∇·u)/∂x via forward finite difference
    let raw = input.as_slice();
    let mut x_plus_raw = raw.to_vec();
    for row in 0..batch {
        x_plus_raw[row * 3] += eps;
    }
    let input_x_plus = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &x_plus_raw, &backend);
    let div_x_plus = Var::new(compute_divergence_2d(forward_fn.clone(), &input_x_plus)?, false);
    let ddiv_dx = coeus_autograd::scalar_mul(
        &coeus_autograd::sub(&div_x_plus, &div_center_var),
        1.0 / eps,
    );

    // ∂(∇·u)/∂y via forward finite difference
    let mut y_plus_raw = raw.to_vec();
    for row in 0..batch {
        y_plus_raw[row * 3 + 1] += eps;
    }
    let input_y_plus = coeus_tensor::Tensor::from_slice_on(vec![batch, 3], &y_plus_raw, &backend);
    let div_y_plus = Var::new(compute_divergence_2d(forward_fn, &input_y_plus)?, false);
    let ddiv_dy = coeus_autograd::scalar_mul(&coeus_autograd::sub(&div_y_plus, &div_center_var), 1.0 / eps);

    Ok((ddiv_dx.tensor, ddiv_dy.tensor))
}
