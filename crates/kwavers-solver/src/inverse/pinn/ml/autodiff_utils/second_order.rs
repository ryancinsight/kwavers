//! Second-order spatial autodiff utilities: ∂²u/∂xᵢ², ∇²u, and ∇(∇·u).
//!
//! # References
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.

use burn::tensor::{backend::AutodiffBackend, Tensor};

use super::spatial::compute_divergence_2d;

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
///
pub fn compute_second_derivative_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
    spatial_dim: usize,
) -> Result<Tensor<B::InnerBackend, 2>, kwavers_core::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    if !(1..=2).contains(&spatial_dim) {
        return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
            "spatial_dim must be 1 (x) or 2 (y), got {}",
            spatial_dim
        )));
    }

    let eps = 1e-4;
    let batch = input.dims()[0];

    let t_col = input.clone().slice([0..batch, 0..1]);
    let x_col = input.clone().slice([0..batch, 1..2]);
    let y_col = input.clone().slice([0..batch, 2..3]);

    let (input_plus, input_minus) = if spatial_dim == 1 {
        let x_plus = x_col.clone().add_scalar(eps);
        let x_minus = x_col.clone().sub_scalar(eps);
        (
            Tensor::cat(vec![t_col.clone(), x_plus, y_col.clone()], 1),
            Tensor::cat(vec![t_col.clone(), x_minus, y_col.clone()], 1),
        )
    } else {
        let y_plus = y_col.clone().add_scalar(eps);
        let y_minus = y_col.clone().sub_scalar(eps);
        (
            Tensor::cat(vec![t_col.clone(), x_col.clone(), y_plus], 1),
            Tensor::cat(vec![t_col.clone(), x_col.clone(), y_minus], 1),
        )
    };

    let output = forward_fn(input.clone());
    let u = output
        .clone()
        .slice([0..output.dims()[0], output_component..output_component + 1]);

    let output_plus = forward_fn(input_plus);
    let u_plus = output_plus.clone().slice([
        0..output_plus.dims()[0],
        output_component..output_component + 1,
    ]);

    let output_minus = forward_fn(input_minus);
    let u_minus = output_minus.clone().slice([
        0..output_minus.dims()[0],
        output_component..output_component + 1,
    ]);

    let d2u = (u_plus + u_minus - u * 2.0) / (eps * eps);
    Ok(d2u.inner())
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
///
pub fn compute_laplacian_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
) -> Result<Tensor<B::InnerBackend, 2>, kwavers_core::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2> + Clone,
{
    let d2u_dx2 = compute_second_derivative_2d(forward_fn.clone(), input, output_component, 1)?;
    let d2u_dy2 = compute_second_derivative_2d(forward_fn, input, output_component, 2)?;
    Ok(d2u_dx2 + d2u_dy2)
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
/// applied to `compute_divergence_2d` to avoid nested `InnerBackend` tensor arithmetic.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn compute_gradient_of_divergence_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
) -> Result<
    (Tensor<B::InnerBackend, 2>, Tensor<B::InnerBackend, 2>),
    kwavers_core::error::KwaversError,
>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2> + Clone,
{
    let eps = 1e-5;
    let batch = input.dims()[0];

    let div_center = compute_divergence_2d(forward_fn.clone(), input)?;

    // ∂(∇·u)/∂x via forward finite difference
    let mut input_x_plus = input.clone();
    let x_col = input_x_plus.clone().slice([0..batch, 0..1]);
    let x_col_plus = x_col.add_scalar(eps);
    input_x_plus = input_x_plus.slice_assign([0..batch, 0..1], x_col_plus);
    let div_x_plus = compute_divergence_2d(forward_fn.clone(), &input_x_plus)?;
    let ddiv_dx = (div_x_plus - div_center.clone()) / eps;

    // ∂(∇·u)/∂y via forward finite difference
    let mut input_y_plus = input.clone();
    let y_col = input_y_plus.clone().slice([0..batch, 1..2]);
    let y_col_plus = y_col.add_scalar(eps);
    input_y_plus = input_y_plus.slice_assign([0..batch, 1..2], y_col_plus);
    let div_y_plus = compute_divergence_2d(forward_fn, &input_y_plus)?;
    let ddiv_dy = (div_y_plus - div_center) / eps;

    Ok((ddiv_dx, ddiv_dy))
}
