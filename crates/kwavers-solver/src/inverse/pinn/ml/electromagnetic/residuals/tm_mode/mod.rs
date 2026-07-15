use super::constants::EPS_FD_F32;
use super::sources::compute_current_density_z;
use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use crate::inverse::pinn::ml::PinnWave2D;
use coeus_autograd::{add, scalar_add, scalar_mul, scalar_sub, sub, Var};

#[cfg(test)]
mod tests;

// ─── TM-mode Maxwell residuals ─────────────────────────────────────────────────

/// TM-mode Faraday-x residual: R_{Fx} = μ ∂Hx/∂t + ∂Ez/∂y
///
/// ## Theorem — 2D TM Faraday-x
///
/// In 2D TM polarisation (E-polarisation; Ez, Hx, Hy in the x–y plane; Ez in z):
/// ```text
/// R_{Fx} = μ ∂Hx/∂t  +  ∂Ez/∂y  =  0
/// ```
/// This is the x-component of ∇×E = −μ ∂H/∂t. In 2D TM: (∇×E)_x = ∂Ez/∂y.
///
/// **Plane-wave verification.** Ez = A·sin(kx·x − ωt), Hx = 0 (no y variation):
/// ∂Ez/∂y = 0 → R_Fx = μ·0 + 0 = 0. ✓
///
/// # References — Jackson (1999) §6.2; Pozar (2011) §1.3.
pub fn tm_mode_faraday_x_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model_ez: &PinnWave2D<B>,
    model_hx: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    t: &Var<f32, B>,
    mu: f64,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let h = EPS_FD_F32;

    // μ ∂Hx/∂t
    let hx_tp = model_hx.forward(x, y, &scalar_add(t, h));
    let hx_tm = model_hx.forward(x, y, &scalar_sub(t, h));
    let dhx_dt = scalar_mul(&sub(&hx_tp, &hx_tm), 1.0 / (2.0 * h));

    // ∂Ez/∂y
    let ez_yp = model_ez.forward(x, &scalar_add(y, h), t);
    let ez_ym = model_ez.forward(x, &scalar_sub(y, h), t);
    let dez_dy = scalar_mul(&sub(&ez_yp, &ez_ym), 1.0 / (2.0 * h));

    add(&scalar_mul(&dhx_dt, mu as f32), &dez_dy)
}

/// TM-mode Faraday-y residual: R_{Fy} = μ ∂Hy/∂t − ∂Ez/∂x
///
/// ## Theorem — 2D TM Faraday-y
///
/// The y-component of ∇×E = −μ ∂H/∂t in 2D TM. In 2D: (∇×E)_y = −∂Ez/∂x.
/// ```text
/// R_{Fy} = μ ∂Hy/∂t  −  ∂Ez/∂x  =  0
/// ```
///
/// **Plane-wave verification.** Ez = A·sin(kx−ωt), Hy = −(Ak/(ωμ))·sin(kx−ωt):
/// μ·∂Hy/∂t = μ·(Akω/(ωμ))·cos(kx−ωt) = Ak·cos(kx−ωt).
/// ∂Ez/∂x = Ak·cos(kx−ωt). R_Fy = 0 ✓
///
/// # References — Jackson (1999) §6.2; Pozar (2011) §1.3.
pub fn tm_mode_faraday_y_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model_ez: &PinnWave2D<B>,
    model_hy: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    t: &Var<f32, B>,
    mu: f64,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let h = EPS_FD_F32;

    // μ ∂Hy/∂t
    let hy_tp = model_hy.forward(x, y, &scalar_add(t, h));
    let hy_tm = model_hy.forward(x, y, &scalar_sub(t, h));
    let dhy_dt = scalar_mul(&sub(&hy_tp, &hy_tm), 1.0 / (2.0 * h));

    // ∂Ez/∂x
    let ez_xp = model_ez.forward(&scalar_add(x, h), y, t);
    let ez_xm = model_ez.forward(&scalar_sub(x, h), y, t);
    let dez_dx = scalar_mul(&sub(&ez_xp, &ez_xm), 1.0 / (2.0 * h));

    sub(&scalar_mul(&dhy_dt, mu as f32), &dez_dx)
}

/// TM-mode Ampère-z residual: R_{Az} = ε ∂Ez/∂t − ∂Hy/∂x + ∂Hx/∂y + σ Ez + Jz
///
/// ## Theorem — 2D TM Ampère-z
///
/// The z-component of Ampère's law ∇×H = σE + ε ∂E/∂t + J in 2D TM:
/// ```text
/// (∇×H)_z = ∂Hy/∂x  −  ∂Hx/∂y
/// ```
/// so the residual is:
/// ```text
/// R_{Az} = ε ∂Ez/∂t  −  ∂Hy/∂x  +  ∂Hx/∂y  +  σ Ez  +  J_ext,z  =  0
/// ```
///
/// **Plane-wave verification (σ=0, J=0).** With Ez = A·sin(kx−ωt),
/// Hy = −(Ak/(ωμ))·sin(kx−ωt), Hx = 0, and dispersion k = ω√(εμ):
///   ε·∂Ez/∂t = −εωA·cos(kx−ωt)
///   −∂Hy/∂x  = (Ak²/(ωμ))·cos(kx−ωt) = εωA·cos(kx−ωt)  [using k²=εμω²]
///   R_Az = 0  ✓
///
/// # References
/// - Jackson (1999) §6.3; Taflove & Hagness (2005) §3.4; Raissi et al. (2019).
// Independent per-component models, field tensors, and parameters with no
// cohesive sub-grouping; bundling would not clarify the call site.
#[allow(clippy::too_many_arguments)]
pub fn tm_mode_ampere_z_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model_ez: &PinnWave2D<B>,
    model_hx: &PinnWave2D<B>,
    model_hy: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    t: &Var<f32, B>,
    eps: f64,
    sigma: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let h = EPS_FD_F32;

    // ε ∂Ez/∂t
    let ez_tp = model_ez.forward(x, y, &scalar_add(t, h));
    let ez_tm = model_ez.forward(x, y, &scalar_sub(t, h));
    let dez_dt = scalar_mul(&sub(&ez_tp, &ez_tm), 1.0 / (2.0 * h));

    // −∂Hy/∂x
    let hy_xp = model_hy.forward(&scalar_add(x, h), y, t);
    let hy_xm = model_hy.forward(&scalar_sub(x, h), y, t);
    let dhy_dx = scalar_mul(&sub(&hy_xp, &hy_xm), 1.0 / (2.0 * h));

    // +∂Hx/∂y
    let hx_yp = model_hx.forward(x, &scalar_add(y, h), t);
    let hx_ym = model_hx.forward(x, &scalar_sub(y, h), t);
    let dhx_dy = scalar_mul(&sub(&hx_yp, &hx_ym), 1.0 / (2.0 * h));

    // σ Ez
    let ez = model_ez.forward(x, y, t);

    // J_ext,z
    let j_z = compute_current_density_z(x, y, physics_params);

    let mut residual = scalar_mul(&dez_dt, eps as f32);
    residual = sub(&residual, &dhy_dx);
    residual = add(&residual, &dhx_dy);
    residual = add(&residual, &scalar_mul(&ez, sigma as f32));
    add(&residual, &j_z)
}
