use super::constants::EPS_FD_F32;
use super::sources::compute_current_density_z;
use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use crate::inverse::pinn::ml::BurnPINN2DWave;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

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
pub fn tm_mode_faraday_x_residual<B: AutodiffBackend>(
    model_ez: &BurnPINN2DWave<B>,
    model_hx: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    t: &Tensor<B, 2>,
    mu: f64,
) -> Tensor<B, 2> {
    let h = EPS_FD_F32;

    // μ ∂Hx/∂t
    let hx_tp = model_hx.forward(x.clone(), y.clone(), t.clone().add_scalar(h));
    let hx_tm = model_hx.forward(x.clone(), y.clone(), t.clone().sub_scalar(h));
    let dhx_dt = (hx_tp.sub(hx_tm)).div_scalar(2.0 * h);

    // ∂Ez/∂y
    let ez_yp = model_ez.forward(x.clone(), y.clone().add_scalar(h), t.clone());
    let ez_ym = model_ez.forward(x.clone(), y.clone().sub_scalar(h), t.clone());
    let dez_dy = (ez_yp.sub(ez_ym)).div_scalar(2.0 * h);

    dhx_dt.mul_scalar(mu as f32).add(dez_dy)
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
pub fn tm_mode_faraday_y_residual<B: AutodiffBackend>(
    model_ez: &BurnPINN2DWave<B>,
    model_hy: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    t: &Tensor<B, 2>,
    mu: f64,
) -> Tensor<B, 2> {
    let h = EPS_FD_F32;

    // μ ∂Hy/∂t
    let hy_tp = model_hy.forward(x.clone(), y.clone(), t.clone().add_scalar(h));
    let hy_tm = model_hy.forward(x.clone(), y.clone(), t.clone().sub_scalar(h));
    let dhy_dt = (hy_tp.sub(hy_tm)).div_scalar(2.0 * h);

    // ∂Ez/∂x
    let ez_xp = model_ez.forward(x.clone().add_scalar(h), y.clone(), t.clone());
    let ez_xm = model_ez.forward(x.clone().sub_scalar(h), y.clone(), t.clone());
    let dez_dx = (ez_xp.sub(ez_xm)).div_scalar(2.0 * h);

    dhy_dt.mul_scalar(mu as f32).sub(dez_dx)
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
pub fn tm_mode_ampere_z_residual<B: AutodiffBackend>(
    model_ez: &BurnPINN2DWave<B>,
    model_hx: &BurnPINN2DWave<B>,
    model_hy: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    t: &Tensor<B, 2>,
    eps: f64,
    sigma: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Tensor<B, 2> {
    let h = EPS_FD_F32;

    // ε ∂Ez/∂t
    let ez_tp = model_ez.forward(x.clone(), y.clone(), t.clone().add_scalar(h));
    let ez_tm = model_ez.forward(x.clone(), y.clone(), t.clone().sub_scalar(h));
    let dez_dt = (ez_tp.sub(ez_tm)).div_scalar(2.0 * h);

    // −∂Hy/∂x
    let hy_xp = model_hy.forward(x.clone().add_scalar(h), y.clone(), t.clone());
    let hy_xm = model_hy.forward(x.clone().sub_scalar(h), y.clone(), t.clone());
    let dhy_dx = (hy_xp.sub(hy_xm)).div_scalar(2.0 * h);

    // +∂Hx/∂y
    let hx_yp = model_hx.forward(x.clone(), y.clone().add_scalar(h), t.clone());
    let hx_ym = model_hx.forward(x.clone(), y.clone().sub_scalar(h), t.clone());
    let dhx_dy = (hx_yp.sub(hx_ym)).div_scalar(2.0 * h);

    // σ Ez
    let ez = model_ez.forward(x.clone(), y.clone(), t.clone());

    // J_ext,z
    let j_z = compute_current_density_z(x, y, physics_params);

    dez_dt
        .mul_scalar(eps as f32)
        .sub(dhy_dx)
        .add(dhx_dy)
        .add(ez.mul_scalar(sigma as f32))
        .add(j_z)
}
