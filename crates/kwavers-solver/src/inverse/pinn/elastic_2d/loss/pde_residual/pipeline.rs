//! End-to-end PDE residual pipelines for 2D elastic wave equation.

#[cfg(feature = "pinn")]
use super::divergence::compute_stress_divergence;
#[cfg(feature = "pinn")]
use super::gradients::compute_displacement_gradients;
#[cfg(feature = "pinn")]
use super::strain_stress::{compute_strain_from_gradients, compute_stress_from_strain};
#[cfg(feature = "pinn")]
use super::time::compute_time_derivatives;
#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Full pipeline: displacement → gradients → strain → stress → divergence.
///
/// Convenience function chaining all autodiff operations for computing ∇·σ.
///
/// ## Returns
///
/// `(div_x, div_y)` — stress divergence components \[N, 1\]
#[cfg(feature = "pinn")]
pub fn displacement_to_stress_divergence<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    lambda: f64,
    mu: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let (dudx, dudy, dvdx, dvdy) = compute_displacement_gradients(u, v, x.clone(), y.clone());
    let (epsilon_xx, epsilon_yy, epsilon_xy) =
        compute_strain_from_gradients(dudx, dudy, dvdx, dvdy);
    let (sigma_xx, sigma_yy, sigma_xy) =
        compute_stress_from_strain(epsilon_xx, epsilon_yy, epsilon_xy, lambda, mu);
    compute_stress_divergence(sigma_xx, sigma_xy, sigma_yy, x, y)
}

/// Compute elastic wave equation PDE residual: R = ρ ∂²u/∂t² − ∇·σ
///
/// The 2D elastic wave equation in displacement form:
/// - ρ ∂²u/∂t² = ∇·σ
///
/// Where σ is computed from displacement via:
/// - strain: ε = ∇_s u (symmetric gradient)
/// - stress: σ = λ tr(ε) I + 2μ ε (Hooke's law)
///
/// ## Arguments
///
/// * `u` - x-displacement \[N, 1\]
/// * `v` - y-displacement \[N, 1\]
/// * `x`, `y`, `t` - Coordinates \[N, 1\] with autodiff enabled
/// * `rho` - Density (kg/m³)
/// * `lambda` - First Lamé parameter (Pa)
/// * `mu` - Shear modulus (Pa)
///
/// ## Returns
///
/// `(residual_x, residual_y)` — PDE residuals per component \[N, 1\]
#[cfg(feature = "pinn")]
pub fn compute_elastic_wave_pde_residual<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    t: Tensor<B, 2>,
    rho: f64,
    lambda: f64,
    mu: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let (_, acceleration) = compute_time_derivatives(Tensor::cat(vec![u.clone(), v.clone()], 1), t);

    let accel_u = acceleration
        .clone()
        .slice([0..acceleration.clone().dims()[0], 0..1]);
    let accel_v = acceleration
        .clone()
        .slice([0..acceleration.dims()[0], 1..2]);

    let (div_sigma_x, div_sigma_y) = displacement_to_stress_divergence(u, v, x, y, lambda, mu);

    let device = div_sigma_x.device();
    let rho_tensor = Tensor::from_floats([rho as f32], &device);

    let residual_x = rho_tensor.clone() * accel_u - div_sigma_x;
    let residual_y = rho_tensor * accel_v - div_sigma_y;

    (residual_x, residual_y)
}
