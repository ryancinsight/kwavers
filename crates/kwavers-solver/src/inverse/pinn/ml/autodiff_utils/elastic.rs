//! Elastic continuum mechanics autodiff utilities: strain tensor and PDE residual.
//!
//! # References
//! - Achenbach, J.D. (1973): *Wave Propagation in Elastic Solids*. North-Holland.
//! - Raissi et al. (2019): "Physics-informed neural networks". *J. Comput. Phys.*, 378, 686–707.

use coeus_autograd::Var;

use super::second_order::{compute_gradient_of_divergence_2d, compute_laplacian_2d};
use super::spatial::compute_spatial_gradient_2d;
use super::time::compute_second_time_derivative;

/// 2D infinitesimal strain components `(ε_xx, ε_yy, ε_xy)`, each `[batch, 1]`.
type StrainTensor2D<B> = (
    coeus_tensor::Tensor<f32, B>,
    coeus_tensor::Tensor<f32, B>,
    coeus_tensor::Tensor<f32, B>,
);

/// 2D elastic-wave PDE residual components `(residual_x, residual_y)`, each `[batch, 1]`.
type ElasticResidual2D<B> = (coeus_tensor::Tensor<f32, B>, coeus_tensor::Tensor<f32, B>);

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
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_strain_tensor_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
) -> Result<StrainTensor2D<B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B> + Clone,
{
    let (du_x_dx, du_x_dy) = compute_spatial_gradient_2d(forward_fn.clone(), input, 0)?;
    let (du_y_dx, du_y_dy) = compute_spatial_gradient_2d(forward_fn, input, 1)?;

    let epsilon_xx = du_x_dx;
    let epsilon_yy = du_y_dy;
    let epsilon_xy = coeus_autograd::scalar_mul(
        &coeus_autograd::add(&Var::new(du_x_dy, false), &Var::new(du_y_dx, false)),
        0.5,
    )
    .tensor;

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
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
pub fn compute_elastic_wave_residual_2d<B, F>(
    forward_fn: F,
    input: &coeus_tensor::Tensor<f32, B>,
    rho: f64,
    lambda: f64,
    mu: f64,
) -> Result<ElasticResidual2D<B>, kwavers_core::error::KwaversError>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    F: Fn(&Var<f32, B>) -> Var<f32, B> + Clone,
{
    let rho = rho as f32;
    let p_wave_coeff = (lambda + 2.0 * mu) as f32;
    let mu = mu as f32;

    // Inertia: ρ ∂²u/∂t²
    let d2u_x_dt2 = compute_second_time_derivative(forward_fn.clone(), input, 0)?;
    let d2u_y_dt2 = compute_second_time_derivative(forward_fn.clone(), input, 1)?;
    let accel_x = coeus_autograd::scalar_mul(&Var::new(d2u_x_dt2, false), rho);
    let accel_y = coeus_autograd::scalar_mul(&Var::new(d2u_y_dt2, false), rho);

    // P-wave term: (λ + 2μ)∇(∇·u)
    let (d_div_dx, d_div_dy) = compute_gradient_of_divergence_2d(forward_fn.clone(), input)?;
    let p_wave_x = coeus_autograd::scalar_mul(&Var::new(d_div_dx, false), p_wave_coeff);
    let p_wave_y = coeus_autograd::scalar_mul(&Var::new(d_div_dy, false), p_wave_coeff);

    // S-wave term: μ∇²u
    let lap_u_x = compute_laplacian_2d(forward_fn.clone(), input, 0)?;
    let lap_u_y = compute_laplacian_2d(forward_fn, input, 1)?;
    let s_wave_x = coeus_autograd::scalar_mul(&Var::new(lap_u_x, false), mu);
    let s_wave_y = coeus_autograd::scalar_mul(&Var::new(lap_u_y, false), mu);

    let residual_x = coeus_autograd::sub(&coeus_autograd::sub(&accel_x, &p_wave_x), &s_wave_x);
    let residual_y = coeus_autograd::sub(&coeus_autograd::sub(&accel_y, &p_wave_y), &s_wave_y);

    Ok((residual_x.tensor, residual_y.tensor))
}
