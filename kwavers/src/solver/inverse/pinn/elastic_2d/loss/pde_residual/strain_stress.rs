//! Strain and stress tensor computation for 2D linear elasticity.

#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute strain tensor components from displacement gradients.
///
/// Linear kinematic relations:
/// - ε_xx = ∂u/∂x
/// - ε_yy = ∂v/∂y
/// - ε_xy = 0.5 * (∂u/∂y + ∂v/∂x)
#[cfg(feature = "pinn")]
pub fn compute_strain_from_gradients<B: AutodiffBackend>(
    dudx: Tensor<B, 2>,
    dudy: Tensor<B, 2>,
    dvdx: Tensor<B, 2>,
    dvdy: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let epsilon_xx = dudx;
    let epsilon_yy = dvdy;
    let epsilon_xy = (dudy + dvdx) * 0.5;

    (epsilon_xx, epsilon_yy, epsilon_xy)
}

/// Compute stress tensor from strain using Hooke's law (isotropic linear elasticity).
///
/// Plane-stress constitutive relations:
/// - σ_xx = (λ + 2μ) ε_xx + λ ε_yy
/// - σ_yy = λ ε_xx + (λ + 2μ) ε_yy
/// - σ_xy = 2μ ε_xy
///
/// Where λ and μ are the Lamé parameters.
#[cfg(feature = "pinn")]
pub fn compute_stress_from_strain<B: AutodiffBackend>(
    epsilon_xx: Tensor<B, 2>,
    epsilon_yy: Tensor<B, 2>,
    epsilon_xy: Tensor<B, 2>,
    lambda: f64,
    mu: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let device = epsilon_xx.device();
    let lambda_tensor = Tensor::from_floats([[lambda as f32]], &device);
    let mu_tensor = Tensor::from_floats([[mu as f32]], &device);
    let two_mu = mu_tensor.clone() * 2.0;

    let sigma_xx = (lambda_tensor.clone() + two_mu.clone()) * epsilon_xx.clone()
        + lambda_tensor.clone() * epsilon_yy.clone();

    let sigma_yy =
        lambda_tensor.clone() * epsilon_xx + (lambda_tensor.clone() + two_mu.clone()) * epsilon_yy;

    let sigma_xy = two_mu * epsilon_xy;

    (sigma_xx, sigma_yy, sigma_xy)
}
