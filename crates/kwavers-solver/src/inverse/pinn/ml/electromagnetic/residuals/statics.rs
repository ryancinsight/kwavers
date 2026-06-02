use super::constants::EPS_FD_F32;
use super::sources::{compute_charge_density, compute_current_density_z};
use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use crate::inverse::pinn::ml::BurnPINN2DWave;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

/// Compute electrostatic residual: ∇·(ε∇φ) = -ρ
pub fn electrostatic_residual<B: AutodiffBackend>(
    model: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    eps: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Tensor<B, 2> {
    // Create input tensor for neural network
    let _inputs = Tensor::cat(vec![x.clone(), y.clone(), Tensor::zeros_like(x)], 1);

    // Forward pass through model to get electric potential φ
    let _phi = model.forward(x.clone(), y.clone(), Tensor::zeros_like(x));

    // Use finite differences within autodiff framework
    let eps_fd = EPS_FD_F32;

    // Compute ∂φ/∂x using central difference
    let x_plus = x.clone().add_scalar(eps_fd);
    let x_minus = x.clone().sub_scalar(eps_fd);
    let phi_x_plus = model.forward(x_plus, y.clone(), Tensor::zeros_like(x));
    let phi_x_minus = model.forward(x_minus, y.clone(), Tensor::zeros_like(x));
    let dphi_dx = (phi_x_plus - phi_x_minus).div_scalar(2.0 * eps_fd);

    // Compute ∂φ/∂y using central difference
    let y_plus = y.clone().add_scalar(eps_fd);
    let y_minus = y.clone().sub_scalar(eps_fd);
    let phi_y_plus = model.forward(x.clone(), y_plus, Tensor::zeros_like(x));
    let phi_y_minus = model.forward(x.clone(), y_minus, Tensor::zeros_like(x));
    let dphi_dy = (phi_y_plus - phi_y_minus).div_scalar(2.0 * eps_fd);

    // Compute D = εE, E = -∇φ
    let _d_x = dphi_dx.mul_scalar(-eps as f32);
    let _d_y = dphi_dy.mul_scalar(-eps as f32);

    // Gauss's law: ∇·D = ρ_free
    let d_x_plus = (model
        .forward(
            x.clone().add_scalar(eps_fd),
            y.clone(),
            Tensor::zeros_like(x),
        )
        .sub(model.forward(
            x.clone().sub_scalar(eps_fd),
            y.clone(),
            Tensor::zeros_like(x),
        )))
    .mul_scalar(-eps as f32)
    .div_scalar(2.0 * eps_fd);

    let d_x_minus = (model
        .forward(
            x.clone().sub_scalar(eps_fd),
            y.clone(),
            Tensor::zeros_like(x),
        )
        .sub(model.forward(
            x.clone().add_scalar(eps_fd),
            y.clone(),
            Tensor::zeros_like(x),
        )))
    .mul_scalar(-eps as f32)
    .div_scalar(2.0 * eps_fd);

    let dd_x_dx = (d_x_plus.sub(d_x_minus)).div_scalar(2.0 * eps_fd);

    let d_y_plus = (model
        .forward(
            x.clone(),
            y.clone().add_scalar(eps_fd),
            Tensor::zeros_like(x),
        )
        .sub(model.forward(
            x.clone(),
            y.clone().sub_scalar(eps_fd),
            Tensor::zeros_like(x),
        )))
    .mul_scalar(-eps as f32)
    .div_scalar(2.0 * eps_fd);

    let d_y_minus = (model
        .forward(
            x.clone(),
            y.clone().sub_scalar(eps_fd),
            Tensor::zeros_like(x),
        )
        .sub(model.forward(
            x.clone(),
            y.clone().add_scalar(eps_fd),
            Tensor::zeros_like(x),
        )))
    .mul_scalar(-eps as f32)
    .div_scalar(2.0 * eps_fd);

    let dd_y_dy = (d_y_plus.sub(d_y_minus)).div_scalar(2.0 * eps_fd);

    let gauss_residual = dd_x_dx.add(dd_y_dy);

    let rho = compute_charge_density(x, y, physics_params);

    gauss_residual.add(rho)
}

/// Compute magnetostatic residual: ∇×(ν∇×A) = μ₀J
pub fn magnetostatic_residual<B: AutodiffBackend>(
    model: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    mu: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Tensor<B, 2> {
    let eps_fd = EPS_FD_F32;

    // Bx = -∂Az/∂y
    let y_plus = y.clone().add_scalar(eps_fd);
    let y_minus = y.clone().sub_scalar(eps_fd);
    let az_y_plus = model.forward(x.clone(), y_plus, Tensor::zeros_like(x));
    let az_y_minus = model.forward(x.clone(), y_minus, Tensor::zeros_like(x));
    let daz_dy = (az_y_plus.sub(az_y_minus)).div_scalar(2.0 * eps_fd);
    let _b_x = daz_dy.mul_scalar(-1.0);

    // By = ∂Az/∂x
    let x_plus = x.clone().add_scalar(eps_fd);
    let x_minus = x.clone().sub_scalar(eps_fd);
    let az_x_plus = model.forward(x_plus, y.clone(), Tensor::zeros_like(x));
    let az_x_minus = model.forward(x_minus, y.clone(), Tensor::zeros_like(x));
    let daz_dx = (az_x_plus.sub(az_x_minus)).div_scalar(2.0 * eps_fd);
    let _b_y = daz_dx;

    // ∇×H: ∂Hy/∂x
    let h_y_x_plus = (model
        .forward(
            x.clone().add_scalar(eps_fd),
            y.clone(),
            Tensor::zeros_like(x),
        )
        .sub(model.forward(
            x.clone().sub_scalar(eps_fd),
            y.clone(),
            Tensor::zeros_like(x),
        )))
    .div_scalar(2.0 * eps_fd)
    .div_scalar(mu as f32);

    let h_y_x_minus = (model
        .forward(
            x.clone().sub_scalar(eps_fd),
            y.clone(),
            Tensor::zeros_like(x),
        )
        .sub(model.forward(
            x.clone().add_scalar(eps_fd),
            y.clone(),
            Tensor::zeros_like(x),
        )))
    .div_scalar(2.0 * eps_fd)
    .div_scalar(mu as f32);

    let dh_y_dx = (h_y_x_plus.sub(h_y_x_minus)).div_scalar(2.0 * eps_fd);

    // -∂Hx/∂y
    let h_x_y_plus = (model
        .forward(
            x.clone(),
            y.clone().add_scalar(eps_fd),
            Tensor::zeros_like(x),
        )
        .sub(model.forward(
            x.clone(),
            y.clone().sub_scalar(eps_fd),
            Tensor::zeros_like(x),
        )))
    .div_scalar(2.0 * eps_fd)
    .div_scalar(mu as f32)
    .mul_scalar(-1.0);

    let h_x_y_minus = (model
        .forward(
            x.clone(),
            y.clone().sub_scalar(eps_fd),
            Tensor::zeros_like(x),
        )
        .sub(model.forward(
            x.clone(),
            y.clone().add_scalar(eps_fd),
            Tensor::zeros_like(x),
        )))
    .div_scalar(2.0 * eps_fd)
    .div_scalar(mu as f32)
    .mul_scalar(-1.0);

    let minus_dh_x_dy = (h_x_y_plus.sub(h_x_y_minus)).div_scalar(2.0 * eps_fd);

    let curl_h_z = dh_y_dx.add(minus_dh_x_dy);

    let j_z = compute_current_density_z(x, y, physics_params);

    curl_h_z.sub(j_z)
}

/// Compute quasi-static residual
pub fn quasi_static_residual<B: AutodiffBackend>(
    model: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    t: &Tensor<B, 2>,
    eps: f64,
    mu: f64,
    sigma: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Tensor<B, 2> {
    // Implemented scalar wave/diffusion equation residual for quasi-static regime
    // Assumes model output u represents a scalar field component (e.g. Ez or Az)
    // Equation: ∇²u - μσ(∂u/∂t) - με(∂²u/∂t²) = -μJ

    let eps_fd = EPS_FD_F32;
    let two = 2.0;

    // Center point
    let u = model.forward(x.clone(), y.clone(), t.clone());

    // 1. Compute Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y²

    // ∂²u/∂x²
    let u_x_plus = model.forward(x.clone().add_scalar(eps_fd), y.clone(), t.clone());
    let u_x_minus = model.forward(x.clone().sub_scalar(eps_fd), y.clone(), t.clone());
    let u_xx = u_x_plus
        .add(u_x_minus)
        .sub(u.clone().mul_scalar(two))
        .div_scalar(eps_fd * eps_fd);

    // ∂²u/∂y²
    let u_y_plus = model.forward(x.clone(), y.clone().add_scalar(eps_fd), t.clone());
    let u_y_minus = model.forward(x.clone(), y.clone().sub_scalar(eps_fd), t.clone());
    let u_yy = u_y_plus
        .add(u_y_minus)
        .sub(u.clone().mul_scalar(two))
        .div_scalar(eps_fd * eps_fd);

    let laplacian = u_xx.add(u_yy);

    // 2. Compute time derivatives

    // ∂u/∂t
    let u_t_plus = model.forward(x.clone(), y.clone(), t.clone().add_scalar(eps_fd));
    let u_t_minus = model.forward(x.clone(), y.clone(), t.clone().sub_scalar(eps_fd));
    let u_t = u_t_plus
        .clone()
        .sub(u_t_minus.clone())
        .div_scalar(two * eps_fd);

    // ∂²u/∂t²
    let u_tt = u_t_plus
        .add(u_t_minus)
        .sub(u.clone().mul_scalar(two))
        .div_scalar(eps_fd * eps_fd);

    // 3. Assemble residual
    // LHS = ∇²u - μσ(∂u/∂t) - με(∂²u/∂t²)
    let term_diffusion = u_t.mul_scalar(mu as f32 * sigma as f32);
    let term_wave = u_tt.mul_scalar(mu as f32 * eps as f32);

    let lhs = laplacian.sub(term_diffusion).sub(term_wave);

    // RHS source term: -μJ
    // Residual = LHS + μJ
    let j_z = compute_current_density_z(x, y, physics_params);
    let source_term = j_z.mul_scalar(mu as f32);

    lhs.add(source_term)
}
