//! Elastic continuum mechanics autodiff utilities: strain tensor and PDE residual.
//!
//! # References
//! - Achenbach, J.D. (1973): *Wave Propagation in Elastic Solids*. North-Holland.
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.

use burn::tensor::{backend::AutodiffBackend, Tensor};

use super::second_order::{compute_gradient_of_divergence_2d, compute_laplacian_2d};
use super::spatial::compute_spatial_gradient_2d;
use super::time::compute_second_time_derivative;

/// Compute infinitesimal strain tensor ε = ½(∇u + ∇uᵀ) for a 2D displacement field.
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]`.
///
/// # Returns
/// Tuple `(ε_xx, ε_yy, ε_xy)`, each of shape `[batch, 1]`.
///
/// # Mathematical Note
/// For displacement field u = [u_x, u_y]:
/// ```text
/// ε_xx = ∂u_x/∂x
/// ε_yy = ∂u_y/∂y
/// ε_xy = ½(∂u_x/∂y + ∂u_y/∂x)
/// ```
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
    let (du_x_dx, du_x_dy) = compute_spatial_gradient_2d(forward_fn.clone(), input, 0)?;
    let (du_y_dx, du_y_dy) = compute_spatial_gradient_2d(forward_fn, input, 1)?;

    let epsilon_xx = du_x_dx;
    let epsilon_yy = du_y_dy;
    let epsilon_xy = (du_x_dy + du_y_dx) * 0.5;

    Ok((epsilon_xx, epsilon_yy, epsilon_xy))
}

/// Compute the elastic wave equation PDE residual for a 2D domain.
///
/// # Arguments
/// - `forward_fn`: Forward pass function.
/// - `input`: Input tensor `[batch, 3]`.
/// - `rho`: Mass density ρ (kg/m³).
/// - `lambda`: Lamé first parameter λ (Pa).
/// - `mu`: Shear modulus μ (Pa).
///
/// # Returns
/// Tuple `(residual_x, residual_y)` for each vector component; zero for an exact solution.
///
/// # Mathematical Specification
/// Isotropic elastic wave equation (Achenbach 1973, §1.2):
/// ```text
/// ρ ∂²u/∂t² = (λ + 2μ)∇(∇·u) + μ∇²u
/// ```
/// PDE residual:
/// ```text
/// R = ρ ∂²u/∂t² − (λ + 2μ)∇(∇·u) − μ∇²u
/// ```
/// `R = 0` ⟺ u satisfies the governing equation.
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
    // Inertia: ρ ∂²u/∂t²
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

    let residual_x = accel_x - p_wave_x - s_wave_x;
    let residual_y = accel_y - p_wave_y - s_wave_y;

    Ok((residual_x, residual_y))
}
