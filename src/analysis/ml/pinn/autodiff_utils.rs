//! Burn Autodiff Utilities for PINN Gradient Computation
//!
//! This module centralizes gradient computation patterns for Burn 0.19+ autodiff,
//! reducing code duplication and simplifying future Burn version upgrades.
//!
//! # Gradient Computation Pattern (Burn 0.19+)
//!
//! The correct pattern for extracting gradients in Burn 0.19 is:
//! ```rust,ignore
//! let input = input.require_grad();
//! let output = model.forward(input.clone());
//! let grads = output.backward();
//! let grad_tensor = input.grad(&grads);
//! ```
//!
//! # Higher-Order Derivatives
//!
//! For second derivatives (needed for PDE residuals like ∂²u/∂x²):
//! 1. Mark the input tensor with `.require_grad()`
//! 2. Compute first derivative and mark with `.require_grad()`
//! 3. Compute second derivative from the first
//!
//! # Usage Examples
//!
//! ```no_run
//! # #[cfg(feature = "pinn")]
//! # {
//! use burn::tensor::Tensor;
//! use burn::backend::Autodiff;
//! use burn::backend::NdArray;
//! use kwavers::ml::pinn::autodiff_utils::*;
//!
//! type B = Autodiff<NdArray>;
//!
//! // First-order spatial gradient ∂u/∂x
//! let (grad_x, grad_y) = compute_spatial_gradient_2d::<B>(
//!     &model,
//!     &input,
//!     0, // x component
//! )?;
//!
//! // Second-order derivative ∂²u/∂x²
//! let laplacian_xx = compute_second_derivative_2d::<B>(
//!     &model,
//!     &input,
//!     0, // x component
//!     1, // with respect to x
//! )?;
//! # Ok::<(), kwavers::error::KwaversError>(())
//! # }
//! ```
//!
//! # Mathematical Specifications
//!
//! ## Elastic Wave Equation Derivatives
//!
//! For u(t, x, y) with output [u_x, u_y]:
//! - Time derivative: ∂u/∂t (velocity)
//! - Second time derivative: ∂²u/∂t² (acceleration)
//! - Spatial gradient: ∇u = [∂u/∂x, ∂u/∂y]
//! - Divergence: ∇·u = ∂u_x/∂x + ∂u_y/∂y
//! - Laplacian: ∇²u = [∂²u_x/∂x² + ∂²u_x/∂y², ∂²u_y/∂x² + ∂²u_y/∂y²]
//!
//! ## Wave Equation PDE Residual
//!
//! ```text
//! ρ ∂²u/∂t² = (λ + 2μ)∇(∇·u) + μ∇²u
//! ```
//!
//! This requires:
//! - `∂²u/∂t²`: second time derivative (acceleration)
//! - `∇·u`: divergence (first-order spatial)
//! - `∇(∇·u)`: gradient of divergence (second-order mixed)
//! - `∇²u`: Laplacian (second-order spatial)

use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute first-order time derivative ∂u/∂t
///
/// # Arguments
/// - `model`: Neural network model implementing `forward(input) -> output`
/// - `input`: Input tensor of shape [batch, 3] where columns are [t, x, y]
/// - `output_component`: Which output component to differentiate (0 for u_x, 1 for u_y)
///
/// # Returns
/// Tensor of shape [batch, 1] containing ∂u/∂t for the specified component
///
/// # Example
/// ```no_run
/// # #[cfg(feature = "pinn")]
/// # {
/// use burn::tensor::Tensor;
/// use burn::backend::Autodiff;
/// use burn::backend::NdArray;
/// use kwavers::ml::pinn::autodiff_utils::compute_time_derivative;
///
/// type B = Autodiff<NdArray>;
/// let velocity = compute_time_derivative::<B, _>(&model, &input, 0)?;
/// # Ok::<(), kwavers::error::KwaversError>(())
/// # }
/// ```
pub fn compute_time_derivative<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
) -> Result<Tensor<B::InnerBackend, 2>, crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    // Mark input as requiring gradients
    let input_grad = input.clone().require_grad();

    // Forward pass
    let output = forward_fn(input_grad.clone());

    // Extract the component we want to differentiate
    let component = output
        .clone()
        .slice([0..output.dims()[0], output_component..output_component + 1]);

    // Backward pass
    let grads = component.sum().backward();

    // Extract gradient with respect to time (column 0)
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

/// Compute second-order time derivative ∂²u/∂t²
///
/// # Arguments
/// - `model`: Neural network model
/// - `input`: Input tensor [batch, 3]
/// - `output_component`: Output component index
///
/// # Returns
/// Tensor [batch, 1] containing ∂²u/∂t²
///
/// # Mathematical Note
/// This computes the acceleration from the displacement field, which is needed
/// for the wave equation residual: ρ ∂²u/∂t² = (forces)
///
/// # Implementation Note
/// Uses finite differences for second derivatives as nested autodiff in Burn
/// requires more complex patterns. This is a pragmatic approach for PDE residuals.
pub fn compute_second_time_derivative<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
) -> Result<Tensor<B::InnerBackend, 2>, crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2> + Clone,
{
    // Use finite differences for second derivative
    let eps = 1e-4;

    // u(t+eps)
    let mut input_plus = input.clone();
    let t_col = input_plus.clone().slice([0..input.dims()[0], 0..1]);
    let t_plus = t_col.clone().add_scalar(eps);
    let x_col = input_plus.clone().slice([0..input.dims()[0], 1..2]);
    let y_col = input_plus.clone().slice([0..input.dims()[0], 2..3]);
    input_plus = Tensor::cat(vec![t_plus.clone(), x_col.clone(), y_col.clone()], 1);

    // u(t-eps)
    let mut input_minus = input.clone();
    let t_minus = t_col.sub_scalar(eps);
    input_minus = Tensor::cat(vec![t_minus, x_col.clone(), y_col.clone()], 1);

    // u(t)
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

    // Second derivative: (u(t+eps) - 2*u(t) + u(t-eps)) / eps^2
    let d2u_dt2 = (u_t_plus + u_t_minus - u_t * 2.0) / (eps * eps);

    Ok(d2u_dt2.inner())
}

/// Compute spatial gradient ∂u/∂x and ∂u/∂y for a 2D field
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model
/// - `input`: Input tensor [batch, 3] with columns [t, x, y]
/// - `output_component`: Output component (0 for u_x, 1 for u_y)
///
/// # Returns
/// Tuple (∂u/∂x, ∂u/∂y) each of shape [batch, 1]
///
/// # Example
/// ```no_run
/// # #[cfg(feature = "pinn")]
/// # {
/// use kwavers::ml::pinn::autodiff_utils::compute_spatial_gradient_2d;
/// let forward = |input| model.forward(input);
/// let (du_dx, du_dy) = compute_spatial_gradient_2d::<B, _>(forward, &input, 0)?;
/// # Ok::<(), kwavers::error::KwaversError>(())
/// # }
/// ```
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

    // Extract spatial gradients (columns 1 and 2 for x and y)
    let grad_tensor = input_grad.grad(&grads).ok_or_else(|| {
        crate::error::KwaversError::InternalError("Failed to compute spatial gradient".into())
    })?;

    let dx_grad = grad_tensor.clone().slice([0..input.dims()[0], 1..2]);
    let dy_grad = grad_tensor.slice([0..input.dims()[0], 2..3]);

    Ok((dx_grad, dy_grad))
}

/// Compute divergence ∇·u = ∂u_x/∂x + ∂u_y/∂y
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model
/// - `input`: Input tensor [batch, 3]
///
/// # Returns
/// Tensor [batch, 1] containing divergence field
///
/// # Mathematical Note
/// For displacement field u = [u_x, u_y]:
/// ```text
/// ∇·u = ∂u_x/∂x + ∂u_y/∂y
/// ```
pub fn compute_divergence_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
) -> Result<Tensor<B::InnerBackend, 2>, crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    // ∂u_x/∂x
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

    // ∂u_y/∂y
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

    // Sum for divergence
    let divergence = du_x_dx + du_y_dy;

    Ok(divergence)
}

/// Compute Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y²
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model
/// - `input`: Input tensor [batch, 3]
/// - `output_component`: Output component
///
/// # Returns
/// Tensor [batch, 1] containing Laplacian
///
/// # Mathematical Note
/// ```text
/// ∇²u = ∂²u/∂x² + ∂²u/∂y²
/// ```
pub fn compute_laplacian_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
) -> Result<Tensor<B::InnerBackend, 2>, crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2> + Clone,
{
    let d2u_dx2 = compute_second_derivative_2d(forward_fn.clone(), input, output_component, 1)?;
    let d2u_dy2 = compute_second_derivative_2d(forward_fn, input, output_component, 2)?;

    // Sum for Laplacian
    let laplacian = d2u_dx2 + d2u_dy2;

    Ok(laplacian)
}

/// Compute second derivative ∂²u/∂xᵢ²
/// Compute second-order spatial derivative ∂²u/∂x² or ∂²u/∂y²
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model
/// - `input`: Input tensor [batch, 3]
/// - `output_component`: Output component (0 for u_x, 1 for u_y)
/// - `spatial_dim`: Spatial dimension (1 for x, 2 for y)
///
/// # Returns
/// Tensor [batch, 1] containing second derivative
pub fn compute_second_derivative_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    output_component: usize,
    spatial_dim: usize,
) -> Result<Tensor<B::InnerBackend, 2>, crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    if !(1..=2).contains(&spatial_dim) {
        return Err(crate::error::KwaversError::InvalidInput(format!(
            "spatial_dim must be 1 (x) or 2 (y), got {}",
            spatial_dim
        )));
    }

    // Use finite differences for second spatial derivatives
    let eps = 1e-4;

    let t_col = input.clone().slice([0..input.dims()[0], 0..1]);
    let x_col = input.clone().slice([0..input.dims()[0], 1..2]);
    let y_col = input.clone().slice([0..input.dims()[0], 2..3]);

    // Create perturbed inputs based on spatial_dim
    let (input_plus, input_minus) = if spatial_dim == 1 {
        // Perturb x
        let x_plus = x_col.clone().add_scalar(eps);
        let x_minus = x_col.clone().sub_scalar(eps);
        (
            Tensor::cat(vec![t_col.clone(), x_plus, y_col.clone()], 1),
            Tensor::cat(vec![t_col.clone(), x_minus, y_col.clone()], 1),
        )
    } else {
        // Perturb y
        let y_plus = y_col.clone().add_scalar(eps);
        let y_minus = y_col.clone().sub_scalar(eps);
        (
            Tensor::cat(vec![t_col.clone(), x_col.clone(), y_plus], 1),
            Tensor::cat(vec![t_col.clone(), x_col.clone(), y_minus], 1),
        )
    };

    // Evaluate at perturbed points
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

    // Second derivative: (u(x+eps) - 2*u(x) + u(x-eps)) / eps^2
    let d2u = (u_plus + u_minus - u * 2.0) / (eps * eps);

    Ok(d2u.inner())
}

/// Compute gradient of divergence ∇(∇·u) needed for P-wave term
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model
/// - `input`: Input tensor [batch, 3]
///
/// # Returns
/// Tuple (∂(∇·u)/∂x, ∂(∇·u)/∂y)
///
/// # Mathematical Specification
/// For the elastic wave equation P-wave term:
/// ```text
/// (λ + 2μ)∇(∇·u) = (λ + 2μ)[∂(∇·u)/∂x, ∂(∇·u)/∂y]
/// ```
///
/// # Implementation Note
/// Uses finite differences to compute second-order derivatives (gradient of divergence),
/// avoiding nested autodiff complexities with InnerBackend tensors.
pub fn compute_gradient_of_divergence_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
) -> Result<(Tensor<B::InnerBackend, 2>, Tensor<B::InnerBackend, 2>), crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2> + Clone,
{
    let eps = 1e-5;
    let batch_size = input.dims()[0];

    // Compute divergence at current point
    let div_center = compute_divergence_2d(forward_fn.clone(), input)?;

    // Perturb x: input[:, 0] += eps
    let mut input_x_plus = input.clone();
    let x_col = input_x_plus.clone().slice([0..batch_size, 0..1]);
    let x_col_plus = x_col.add_scalar(eps);
    input_x_plus = input_x_plus.slice_assign([0..batch_size, 0..1], x_col_plus);
    let div_x_plus = compute_divergence_2d(forward_fn.clone(), &input_x_plus)?;

    // ∂(∇·u)/∂x via finite difference
    let ddiv_dx = (div_x_plus - div_center.clone()) / eps;

    // Perturb y: input[:, 1] += eps
    let mut input_y_plus = input.clone();
    let y_col = input_y_plus.clone().slice([0..batch_size, 1..2]);
    let y_col_plus = y_col.add_scalar(eps);
    input_y_plus = input_y_plus.slice_assign([0..batch_size, 1..2], y_col_plus);
    let div_y_plus = compute_divergence_2d(forward_fn, &input_y_plus)?;

    // ∂(∇·u)/∂y via finite difference
    let ddiv_dy = (div_y_plus - div_center) / eps;

    Ok((ddiv_dx, ddiv_dy))
}

/// Compute strain tensor ε = (1/2)(∇u + ∇uᵀ) for 2D
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model
/// - `input`: Input tensor [batch, 3]
///
/// # Returns
/// Tuple (ε_xx, ε_yy, ε_xy) each of shape [batch, 1]
///
/// # Mathematical Note
/// For displacement field u = [u_x, u_y]:
/// - ε_xx = ∂u_x/∂x
/// - ε_yy = ∂u_y/∂y
/// - ε_xy = (1/2)(∂u_x/∂y + ∂u_y/∂x)
pub fn compute_strain_tensor_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
) -> Result<
    (
        Tensor<B::InnerBackend, 2>,
        Tensor<B::InnerBackend, 2>,
        Tensor<B::InnerBackend, 2>,
    ),
    crate::error::KwaversError,
>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2> + Clone,
{
    // Compute gradients of u_x
    let (du_x_dx, du_x_dy) = compute_spatial_gradient_2d(forward_fn.clone(), input, 0)?;

    // Compute gradients of u_y
    let (du_y_dx, du_y_dy) = compute_spatial_gradient_2d(forward_fn, input, 1)?;

    // Strain tensor components
    let epsilon_xx = du_x_dx;
    let epsilon_yy = du_y_dy;
    let epsilon_xy = (du_x_dy + du_y_dx) * 0.5;

    Ok((epsilon_xx, epsilon_yy, epsilon_xy))
}

/// Compute full elastic wave equation PDE residual for 2D
///
/// # Arguments
/// - `forward_fn`: Function that performs forward pass through the model
/// - `input`: Input tensor [batch, 3]
/// - `rho`: Density (kg/m³)
/// - `lambda`: Lamé first parameter (Pa)
/// - `mu`: Shear modulus (Pa)
///
/// # Returns
/// Tuple (residual_x, residual_y) for each component of the wave equation
///
/// # Mathematical Specification
/// ```text
/// ρ ∂²u/∂t² = (λ + 2μ)∇(∇·u) + μ∇²u
/// ```
///
/// Residual = ρ ∂²u/∂t² - (λ + 2μ)∇(∇·u) - μ∇²u
///
/// Should be zero for a valid solution.
pub fn compute_elastic_wave_residual_2d<B, F>(
    forward_fn: F,
    input: &Tensor<B, 2>,
    rho: f64,
    lambda: f64,
    mu: f64,
) -> Result<(Tensor<B::InnerBackend, 2>, Tensor<B::InnerBackend, 2>), crate::error::KwaversError>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2> + Clone,
{
    // Acceleration terms: ρ ∂²u/∂t²
    let d2u_x_dt2 = compute_second_time_derivative(forward_fn.clone(), input, 0)?;
    let d2u_y_dt2 = compute_second_time_derivative(forward_fn.clone(), input, 1)?;
    let accel_x = d2u_x_dt2 * rho;
    let accel_y = d2u_y_dt2 * rho;

    // P-wave term: (λ + 2μ)∇(∇·u)
    let (d_div_dx, d_div_dy) = compute_gradient_of_divergence_2d(forward_fn.clone(), input)?;
    let p_wave_coeff = lambda + 2.0 * mu;
    let p_wave_x = d_div_dx * p_wave_coeff;
    let p_wave_y = d_div_dy * p_wave_coeff;

    // S-wave term: μ∇²u
    let lap_u_x = compute_laplacian_2d(forward_fn.clone(), input, 0)?;
    let lap_u_y = compute_laplacian_2d(forward_fn, input, 1)?;
    let s_wave_x = lap_u_x * mu;
    let s_wave_y = lap_u_y * mu;

    // PDE residual: ρ ∂²u/∂t² - (λ + 2μ)∇(∇·u) - μ∇²u
    let residual_x = accel_x - p_wave_x - s_wave_x;
    let residual_y = accel_y - p_wave_y - s_wave_y;

    Ok((residual_x, residual_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn test_time_derivative_matches_analytic() -> crate::core::error::KwaversResult<()> {
        type B = Autodiff<NdArray<f32>>;

        let device = Default::default();

        let input = Tensor::<B, 2>::from_floats([[0.2, 0.4, 0.6], [0.7, 0.1, 0.9]], &device);

        let forward = |input: Tensor<B, 2>| {
            let batch = input.dims()[0];
            let t = input.clone().slice([0..batch, 0..1]);
            let x = input.clone().slice([0..batch, 1..2]);
            let y = input.slice([0..batch, 2..3]);

            let u0 = t.clone().powf_scalar(2.0) + x.clone() + y.clone();
            let u1 = t.clone() * 2.0 + x * 3.0 - y;

            Tensor::cat(vec![u0, u1], 1)
        };

        let du0_dt = compute_time_derivative::<B, _>(forward, &input, 0)?;
        let data = du0_dt.into_data();
        let values = data.as_slice::<f32>().map_err(|e| {
            crate::core::error::KwaversError::System(
                crate::core::error::SystemError::InvalidOperation {
                    operation: "tensor_to_f32_slice".to_string(),
                    reason: format!("{e:?}"),
                },
            )
        })?;

        let expected = [0.4_f32, 1.4_f32];
        for (got, exp) in values.iter().copied().zip(expected) {
            assert!((got - exp).abs() < 1e-4);
        }

        Ok(())
    }
}
