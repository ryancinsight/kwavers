//! PDE residual computation using automatic differentiation
//!
//! This module implements the complete chain of automatic differentiation
//! operations required to compute the elastic wave equation PDE residual.
//!
//! ## Mathematical Pipeline
//!
//! The elastic wave equation in 2D displacement form is:
//!     ρ ∂²u/∂t² = ∇·σ
//!
//! Where:
//!   - u = (u, v) is the displacement vector field
//!   - ρ is the material density
//!   - σ is the Cauchy stress tensor
//!   - ∇·σ is the divergence of the stress tensor
//!
//! ## Computational Chain (All via Autodiff)
//!
//! 1. **Displacement Gradients**: compute_displacement_gradients()
//!    Input:  u(x,y,t), v(x,y,t) from neural network
//!    Output: ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y
//!    Method: Burn autodiff (backward pass on network outputs)
//!
//! 2. **Strain Tensor**: compute_strain_from_gradients()
//!    Input:  Displacement gradients from step 1
//!    Output: ε_xx, ε_yy, ε_xy
//!    Method: Linear kinematic relations
//!
//! ```text
//! ε_xx = ∂u/∂x
//! ε_yy = ∂v/∂y
//! ε_xy = 0.5(∂u/∂y + ∂v/∂x)
//! ```
//!
//! 3. **Stress Tensor**: compute_stress_from_strain()
//!    Input:  Strain components from step 2
//!    Output: σ_xx, σ_yy, σ_xy
//!    Method: Hooke's law (isotropic linear elasticity)
//!
//! ```text
//! σ_xx = (λ + 2μ) ε_xx + λ ε_yy
//! σ_yy = λ ε_xx + (λ + 2μ) ε_yy
//! σ_xy = 2μ ε_xy
//! ```
//!
//! 4. **Stress Divergence**: compute_stress_divergence()
//!    Input:  Stress components from step 3
//!    Output: ∂σ_xx/∂x + ∂σ_xy/∂y, ∂σ_xy/∂x + ∂σ_yy/∂y
//!    Method: Burn autodiff (backward pass on stress tensors)
//!
//! 5. **Time Derivatives**: compute_time_derivatives()
//!    Input:  u(x,y,t) from neural network
//!    Output: ∂u/∂t (velocity), ∂²u/∂t² (acceleration)
//!    Method: Burn autodiff (two passes: first for ∂/∂t, second for ∂²/∂t²)
//!
//! 6. **PDE Residual**: compute_elastic_wave_pde_residual()
//!    Input:  Network outputs u, v and coordinates x, y, t
//!    Output: R = ρ ∂²u/∂t² - ∇·σ
//!    Method: Combines steps 1-5

#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute spatial gradients of displacement using automatic differentiation
///
/// This computes ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y using Burn's autodiff capabilities.
///
/// # Mathematical Foundation
///
/// Given displacement field u(x,y), we compute spatial derivatives via
/// backpropagation through the network with respect to input coordinates.
///
/// # Arguments
///
/// * `u` - x-displacement [N, 1]
/// * `v` - y-displacement [N, 1]
/// * `x` - x-coordinates [N, 1] (must support autodiff)
/// * `y` - y-coordinates [N, 1] (must support autodiff)
///
/// # Returns
///
/// * (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)
///
/// # Implementation Note
///
/// This function computes first-order derivatives using Burn's gradient computation.
/// The inputs x and y must be tensors that track gradients (i.e., from the forward pass).
#[cfg(feature = "pinn")]
pub fn compute_displacement_gradients<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    // Compute ∂u/∂x using backward pass
    let grads = u.clone().backward();
    let dudx = x
        .grad(&grads)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| x.zeros_like());

    // Compute ∂u/∂y
    let grads_u_y = u.backward();
    let dudy = y
        .grad(&grads_u_y)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| y.zeros_like());

    // Compute ∂v/∂x
    let grads_v_x = v.clone().backward();
    let dvdx = x
        .grad(&grads_v_x)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| x.zeros_like());

    // Compute ∂v/∂y
    let grads_v_y = v.backward();
    let dvdy = y
        .grad(&grads_v_y)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| y.zeros_like());

    (dudx, dudy, dvdx, dvdy)
}

/// Compute strain tensor components from displacement gradients
///
/// # Mathematical Foundation
///
/// Strain tensor (linear elasticity):
/// * ε_xx = ∂u/∂x
/// * ε_yy = ∂v/∂y
/// * ε_xy = 0.5 * (∂u/∂y + ∂v/∂x)
///
/// # Arguments
///
/// * `dudx` - ∂u/∂x [N, 1]
/// * `dudy` - ∂u/∂y [N, 1]
/// * `dvdx` - ∂v/∂x [N, 1]
/// * `dvdy` - ∂v/∂y [N, 1]
///
/// # Returns
///
/// * (ε_xx, ε_yy, ε_xy)
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

/// Compute stress tensor from strain using Hooke's law
///
/// # Mathematical Foundation
///
/// Plane stress (2D elastic):
/// * σ_xx = (λ + 2μ) ε_xx + λ ε_yy
/// * σ_yy = λ ε_xx + (λ + 2μ) ε_yy
/// * σ_xy = 2μ ε_xy
///
/// Where λ and μ are Lamé parameters.
///
/// # Arguments
///
/// * `epsilon_xx` - Normal strain in x
/// * `epsilon_yy` - Normal strain in y
/// * `epsilon_xy` - Shear strain
/// * `lambda` - First Lamé parameter
/// * `mu` - Second Lamé parameter (shear modulus)
///
/// # Returns
///
/// * (σ_xx, σ_yy, σ_xy)
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

    // σ_xx = (λ + 2μ) ε_xx + λ ε_yy
    let sigma_xx = (lambda_tensor.clone() + two_mu.clone()) * epsilon_xx.clone()
        + lambda_tensor.clone() * epsilon_yy.clone();

    // σ_yy = λ ε_xx + (λ + 2μ) ε_yy
    let sigma_yy =
        lambda_tensor.clone() * epsilon_xx + (lambda_tensor.clone() + two_mu.clone()) * epsilon_yy;

    // σ_xy = 2μ ε_xy
    let sigma_xy = two_mu * epsilon_xy;

    (sigma_xx, sigma_yy, sigma_xy)
}

/// Compute stress divergence ∇·σ using automatic differentiation
///
/// # Mathematical Foundation
///
/// Divergence of stress tensor:
/// * div_x = ∂σ_xx/∂x + ∂σ_xy/∂y
/// * div_y = ∂σ_xy/∂x + ∂σ_yy/∂y
///
/// This appears in the elastic wave equation: ρ ∂²u/∂t² = ∇·σ
///
/// # Arguments
///
/// * `sigma_xx` - Stress component σ_xx [N, 1]
/// * `sigma_xy` - Stress component σ_xy [N, 1]
/// * `sigma_yy` - Stress component σ_yy [N, 1]
/// * `x` - x-coordinates [N, 1] (must support autodiff)
/// * `y` - y-coordinates [N, 1] (must support autodiff)
///
/// # Returns
///
/// * (∂σ_xx/∂x + ∂σ_xy/∂y, ∂σ_xy/∂x + ∂σ_yy/∂y)
#[cfg(feature = "pinn")]
pub fn compute_stress_divergence<B: AutodiffBackend>(
    sigma_xx: Tensor<B, 2>,
    sigma_xy: Tensor<B, 2>,
    sigma_yy: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // ∂σ_xx/∂x
    let grads_sxx = sigma_xx.clone().backward();
    let dsxx_dx = x
        .grad(&grads_sxx)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| x.zeros_like());

    let grads_sxy = sigma_xy.clone().backward();
    let dsxy_dy = y
        .grad(&grads_sxy)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| y.zeros_like());
    let dsxy_dx = x
        .grad(&grads_sxy)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| x.zeros_like());

    // ∂σ_yy/∂y
    let grads_syy = sigma_yy.backward();
    let dsyy_dy = y
        .grad(&grads_syy)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| y.zeros_like());

    // Divergence components
    let div_x = dsxx_dx + dsxy_dy;
    let div_y = dsxy_dx + dsyy_dy;

    (div_x, div_y)
}

/// Compute time derivatives ∂u/∂t and ∂²u/∂t² using automatic differentiation
///
/// # Mathematical Foundation
///
/// Velocity: v = ∂u/∂t
/// Acceleration: a = ∂²u/∂t² = ∂v/∂t
///
/// # Arguments
///
/// * `u` - Displacement [N, 2]
/// * `t` - Time coordinates [N, 1] (must support autodiff)
///
/// # Returns
///
/// * (velocity, acceleration) each [N, 2]
///
/// # Note
///
/// For second derivatives, we apply autodiff twice:
/// 1. First pass: compute ∂u/∂t
/// 2. Second pass: compute ∂(∂u/∂t)/∂t
#[cfg(feature = "pinn")]
pub fn compute_time_derivatives<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    t: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // First derivative: velocity = ∂u/∂t
    let grads_u = u.clone().backward();
    let velocity = t
        .grad(&grads_u)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| u.zeros_like());

    // Second derivative: acceleration = ∂²u/∂t² = ∂(∂u/∂t)/∂t
    let grads_velocity = velocity.clone().backward();
    let acceleration = t
        .grad(&grads_velocity)
        .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
        .unwrap_or_else(|| velocity.zeros_like());

    (velocity, acceleration)
}

/// Full pipeline: displacement → gradients → strain → stress → divergence
///
/// This is a convenience function that chains all autodiff operations
/// for computing the PDE residual.
///
/// # Arguments
///
/// * `u` - x-displacement [N, 1]
/// * `v` - y-displacement [N, 1]
/// * `x` - x-coordinates [N, 1]
/// * `y` - y-coordinates [N, 1]
/// * `lambda` - First Lamé parameter
/// * `mu` - Shear modulus
///
/// # Returns
///
/// * (div_x, div_y) - Stress divergence components
#[cfg(feature = "pinn")]
pub fn displacement_to_stress_divergence<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    lambda: f64,
    mu: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // Step 1: Compute displacement gradients
    let (dudx, dudy, dvdx, dvdy) = compute_displacement_gradients(u, v, x.clone(), y.clone());

    // Step 2: Compute strain
    let (epsilon_xx, epsilon_yy, epsilon_xy) =
        compute_strain_from_gradients(dudx, dudy, dvdx, dvdy);

    // Step 3: Compute stress via Hooke's law
    let (sigma_xx, sigma_yy, sigma_xy) =
        compute_stress_from_strain(epsilon_xx, epsilon_yy, epsilon_xy, lambda, mu);

    // Step 4: Compute stress divergence
    compute_stress_divergence(sigma_xx, sigma_xy, sigma_yy, x, y)
}

/// Compute elastic wave equation PDE residual using autodiff
///
/// # Mathematical Foundation
///
/// The 2D elastic wave equation in displacement form:
/// * ρ ∂²u/∂t² = ∇·σ
///
/// Where σ is the stress tensor computed from displacement via:
/// * strain: ε = ∇_s u (symmetric gradient)
/// * stress: σ = λ tr(ε) I + 2μ ε (Hooke's law)
///
/// The PDE residual is: R = ρ ∂²u/∂t² - ∇·σ
///
/// # Arguments
///
/// * `u` - x-displacement [N, 1]
/// * `v` - y-displacement [N, 1]
/// * `x` - x-coordinates [N, 1] (autodiff enabled)
/// * `y` - y-coordinates [N, 1] (autodiff enabled)
/// * `t` - time coordinates [N, 1] (autodiff enabled)
/// * `rho` - Density (kg/m³)
/// * `lambda` - First Lamé parameter (Pa)
/// * `mu` - Shear modulus (Pa)
///
/// # Returns
///
/// * (residual_x, residual_y) - PDE residuals for each component [N, 1]
///
/// # Note
///
/// This function requires that x, y, t tensors support gradient tracking.
/// The network output (u, v) must be differentiable with respect to inputs.
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
    // Compute acceleration: ∂²u/∂t²
    let (_, acceleration) = compute_time_derivatives(Tensor::cat(vec![u.clone(), v.clone()], 1), t);

    // Split acceleration into components
    let accel_u = acceleration
        .clone()
        .slice([0..acceleration.clone().dims()[0], 0..1]);
    let accel_v = acceleration
        .clone()
        .slice([0..acceleration.dims()[0], 1..2]);

    // Compute stress divergence: ∇·σ
    let (div_sigma_x, div_sigma_y) = displacement_to_stress_divergence(u, v, x, y, lambda, mu);

    // Create density tensor
    let device = div_sigma_x.device();
    let rho_tensor = Tensor::from_floats([rho as f32], &device);

    // PDE residual: ρ ∂²u/∂t² - ∇·σ
    let residual_x = rho_tensor.clone() * accel_u - div_sigma_x;
    let residual_y = rho_tensor * accel_v - div_sigma_y;

    (residual_x, residual_y)
}

#[cfg(all(test, feature = "pinn"))]
mod tests {
    use super::*;

    #[cfg(feature = "pinn")]
    #[test]
    fn test_strain_computation_mathematical_properties() -> crate::error::KwaversResult<()> {
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let alpha = 2.5_f32;
        let dudx = Tensor::<B, 2>::from_floats([[alpha]], &device);
        let dudy = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);
        let dvdx = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);
        let dvdy = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);

        let (eps_xx, eps_yy, eps_xy) = compute_strain_from_gradients(dudx, dudy, dvdx, dvdy);

        let eps_xx = eps_xx.into_data().as_slice::<f32>().unwrap()[0];
        let eps_yy = eps_yy.into_data().as_slice::<f32>().unwrap()[0];
        let eps_xy = eps_xy.into_data().as_slice::<f32>().unwrap()[0];

        assert!((eps_xx - alpha).abs() < 1e-6);
        assert!(eps_yy.abs() < 1e-6);
        assert!(eps_xy.abs() < 1e-6);
        Ok(())
    }

    #[cfg(feature = "pinn")]
    #[test]
    fn test_hookes_law_isotropic() -> crate::error::KwaversResult<()> {
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let lambda = 5.0_f64;
        let mu = 3.0_f64;
        let gamma = 1.2_f32;

        let eps_xx = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);
        let eps_yy = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);
        let eps_xy = Tensor::<B, 2>::from_floats([[gamma]], &device);

        let (sigma_xx, sigma_yy, sigma_xy) =
            compute_stress_from_strain(eps_xx, eps_yy, eps_xy, lambda, mu);

        let sigma_xx = sigma_xx.into_data().as_slice::<f32>().unwrap()[0];
        let sigma_yy = sigma_yy.into_data().as_slice::<f32>().unwrap()[0];
        let sigma_xy = sigma_xy.into_data().as_slice::<f32>().unwrap()[0];

        assert!(sigma_xx.abs() < 1e-6);
        assert!(sigma_yy.abs() < 1e-6);
        assert!((sigma_xy - (2.0_f32 * mu as f32 * gamma)).abs() < 1e-6);
        Ok(())
    }

    #[cfg(feature = "pinn")]
    #[test]
    fn test_stress_divergence_equilibrium() -> crate::error::KwaversResult<()> {
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let x = Tensor::<B, 2>::from_floats([[0.1_f32]], &device).require_grad();
        let y = Tensor::<B, 2>::from_floats([[0.3_f32]], &device).require_grad();

        let sigma_xx = x.clone().mul_scalar(2.0);
        let sigma_yy = x.clone().mul_scalar(3.0);
        let sigma_xy = y.clone().mul_scalar(-2.0);

        let (div_x, div_y) = compute_stress_divergence(sigma_xx, sigma_xy, sigma_yy, x, y);

        let div_x_data = div_x.into_data();
        let div_x = div_x_data.as_slice::<f32>().unwrap();

        let div_y_data = div_y.into_data();
        let div_y = div_y_data.as_slice::<f32>().unwrap();

        for &v in div_x.iter() {
            assert!(v.abs() < 1e-6);
        }
        for &v in div_y.iter() {
            assert!(v.abs() < 1e-6);
        }
        Ok(())
    }
}
