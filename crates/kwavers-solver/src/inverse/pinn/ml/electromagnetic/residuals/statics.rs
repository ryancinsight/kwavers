use super::constants::EPS_FD_F32;
use super::sources::{compute_charge_density, compute_current_density_z};
use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use crate::inverse::pinn::ml::PinnWave2D;
use coeus_autograd::{add, scalar_add, scalar_mul, scalar_sub, sub, Var};

/// Compute electrostatic residual: ∇·(ε∇φ) = -ρ
pub fn electrostatic_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    eps: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let eps_fd = EPS_FD_F32;
    let backend = B::default();
    let zero = Var::new(
        coeus_tensor::Tensor::zeros_on(x.tensor.shape(), &backend),
        false,
    );

    // Gauss's law: ∇·D = ρ_free, D = -ε∇φ
    let d_x_plus = scalar_mul(
        &sub(
            &model.forward(&scalar_add(x, eps_fd), y, &zero),
            &model.forward(&scalar_sub(x, eps_fd), y, &zero),
        ),
        -(eps as f32) / (2.0 * eps_fd),
    );
    let d_x_minus = scalar_mul(
        &sub(
            &model.forward(&scalar_sub(x, eps_fd), y, &zero),
            &model.forward(&scalar_add(x, eps_fd), y, &zero),
        ),
        -(eps as f32) / (2.0 * eps_fd),
    );
    let dd_x_dx = scalar_mul(&sub(&d_x_plus, &d_x_minus), 1.0 / (2.0 * eps_fd));

    let d_y_plus = scalar_mul(
        &sub(
            &model.forward(x, &scalar_add(y, eps_fd), &zero),
            &model.forward(x, &scalar_sub(y, eps_fd), &zero),
        ),
        -(eps as f32) / (2.0 * eps_fd),
    );
    let d_y_minus = scalar_mul(
        &sub(
            &model.forward(x, &scalar_sub(y, eps_fd), &zero),
            &model.forward(x, &scalar_add(y, eps_fd), &zero),
        ),
        -(eps as f32) / (2.0 * eps_fd),
    );
    let dd_y_dy = scalar_mul(&sub(&d_y_plus, &d_y_minus), 1.0 / (2.0 * eps_fd));

    let gauss_residual = add(&dd_x_dx, &dd_y_dy);

    let rho = compute_charge_density(x, y, physics_params);

    add(&gauss_residual, &rho)
}

/// Compute magnetostatic residual: ∇×(ν∇×A) = μ₀J
pub fn magnetostatic_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    mu: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let eps_fd = EPS_FD_F32;
    let backend = B::default();
    let zero = Var::new(
        coeus_tensor::Tensor::zeros_on(x.tensor.shape(), &backend),
        false,
    );

    // ∇×H: ∂Hy/∂x
    let h_y_x_plus = scalar_mul(
        &sub(
            &model.forward(&scalar_add(x, eps_fd), y, &zero),
            &model.forward(&scalar_sub(x, eps_fd), y, &zero),
        ),
        1.0 / (2.0 * eps_fd * mu as f32),
    );
    let h_y_x_minus = scalar_mul(
        &sub(
            &model.forward(&scalar_sub(x, eps_fd), y, &zero),
            &model.forward(&scalar_add(x, eps_fd), y, &zero),
        ),
        1.0 / (2.0 * eps_fd * mu as f32),
    );
    let dh_y_dx = scalar_mul(&sub(&h_y_x_plus, &h_y_x_minus), 1.0 / (2.0 * eps_fd));

    // -∂Hx/∂y
    let h_x_y_plus = scalar_mul(
        &sub(
            &model.forward(x, &scalar_add(y, eps_fd), &zero),
            &model.forward(x, &scalar_sub(y, eps_fd), &zero),
        ),
        -1.0 / (2.0 * eps_fd * mu as f32),
    );
    let h_x_y_minus = scalar_mul(
        &sub(
            &model.forward(x, &scalar_sub(y, eps_fd), &zero),
            &model.forward(x, &scalar_add(y, eps_fd), &zero),
        ),
        -1.0 / (2.0 * eps_fd * mu as f32),
    );
    let minus_dh_x_dy = scalar_mul(&sub(&h_x_y_plus, &h_x_y_minus), 1.0 / (2.0 * eps_fd));

    let curl_h_z = add(&dh_y_dx, &minus_dh_x_dy);

    let j_z = compute_current_density_z(x, y, physics_params);

    sub(&curl_h_z, &j_z)
}

/// Compute quasi-static residual
// Independent field tensors and physical parameters with no cohesive
// sub-grouping; bundling would not clarify the call site.
#[allow(clippy::too_many_arguments)]
pub fn quasi_static_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    t: &Var<f32, B>,
    eps: f64,
    mu: f64,
    sigma: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    // Implemented scalar wave/diffusion equation residual for quasi-static regime
    // Assumes model output u represents a scalar field component (e.g. Ez or Az)
    // Equation: ∇²u - μσ(∂u/∂t) - με(∂²u/∂t²) = -μJ

    let eps_fd = EPS_FD_F32;
    let two = 2.0_f32;

    // Center point
    let u = model.forward(x, y, t);

    // 1. Compute Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y²

    // ∂²u/∂x²
    let u_x_plus = model.forward(&scalar_add(x, eps_fd), y, t);
    let u_x_minus = model.forward(&scalar_sub(x, eps_fd), y, t);
    let u_xx = scalar_mul(
        &sub(&add(&u_x_plus, &u_x_minus), &scalar_mul(&u, two)),
        1.0 / (eps_fd * eps_fd),
    );

    // ∂²u/∂y²
    let u_y_plus = model.forward(x, &scalar_add(y, eps_fd), t);
    let u_y_minus = model.forward(x, &scalar_sub(y, eps_fd), t);
    let u_yy = scalar_mul(
        &sub(&add(&u_y_plus, &u_y_minus), &scalar_mul(&u, two)),
        1.0 / (eps_fd * eps_fd),
    );

    let laplacian = add(&u_xx, &u_yy);

    // 2. Compute time derivatives

    // ∂u/∂t
    let u_t_plus = model.forward(x, y, &scalar_add(t, eps_fd));
    let u_t_minus = model.forward(x, y, &scalar_sub(t, eps_fd));
    let u_t = scalar_mul(&sub(&u_t_plus, &u_t_minus), 1.0 / (two * eps_fd));

    // ∂²u/∂t²
    let u_tt = scalar_mul(
        &sub(&add(&u_t_plus, &u_t_minus), &scalar_mul(&u, two)),
        1.0 / (eps_fd * eps_fd),
    );

    // 3. Assemble residual
    // LHS = ∇²u - μσ(∂u/∂t) - με(∂²u/∂t²)
    let term_diffusion = scalar_mul(&u_t, mu as f32 * sigma as f32);
    let term_wave = scalar_mul(&u_tt, mu as f32 * eps as f32);

    let lhs = sub(&sub(&laplacian, &term_diffusion), &term_wave);

    // RHS source term: -μJ
    // Residual = LHS + μJ
    let j_z = compute_current_density_z(x, y, physics_params);
    let source_term = scalar_mul(&j_z, mu as f32);

    add(&lhs, &source_term)
}
