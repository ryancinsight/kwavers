use super::constants::EPS_FD_F32;
use super::sources::compute_charge_density;
use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use crate::inverse::pinn::ml::BurnPINN2DWave;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

// ─── TE-mode Maxwell residuals ─────────────────────────────────────────────────

/// TE-mode Faraday residual: R_F = μ ∂H_z/∂t − ∂E_x/∂y + ∂E_y/∂x
///
/// ## Theorem — 2D TE Faraday's Law
///
/// In 2D TE polarisation (H-polarisation; Hz, Ex, Ey in the x–y plane; Hz in z):
/// ```text
/// R_F = μ₀ μ_r ∂Hz/∂t  −  ∂Ex/∂y  +  ∂Ey/∂x  =  0
/// ```
/// This is the z-component of ∇×E = −μ ∂H/∂t. It is exactly zero for every
/// solution of Maxwell's equations in a linear, isotropic, non-dispersive medium.
///
/// **Plane-wave verification.** For Ex = A·sin(ky·y − ωt), Ey = 0, Hz chosen by
/// Faraday's law: substituting gives εμω² = k² (dispersion), confirming R_F = 0. ✓
///
/// **Algorithm.** Second-order central FD, step `EPS_FD_F32`:
/// ```text
/// ∂Hz/∂t  ≈ (Hz(t+h) − Hz(t−h)) / (2h)
/// ∂Ex/∂y  ≈ (Ex(y+h) − Ex(y−h)) / (2h)
/// ∂Ey/∂x  ≈ (Ey(x+h) − Ey(x−h)) / (2h)
/// ```
///
/// # References
/// - Jackson, J.D. (1999). *Classical Electrodynamics* (3rd ed.). §6.2.
/// - Pozar, D.M. (2011). *Microwave Engineering* (4th ed.). §1.3.
pub fn te_mode_faraday_residual<B: AutodiffBackend>(
    model_ex: &BurnPINN2DWave<B>,
    model_ey: &BurnPINN2DWave<B>,
    model_hz: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    t: &Tensor<B, 2>,
    mu: f64,
) -> Tensor<B, 2> {
    let h = EPS_FD_F32;

    // μ ∂Hz/∂t
    let hz_tp = model_hz.forward(x.clone(), y.clone(), t.clone().add_scalar(h));
    let hz_tm = model_hz.forward(x.clone(), y.clone(), t.clone().sub_scalar(h));
    let dhz_dt = (hz_tp.sub(hz_tm)).div_scalar(2.0 * h);

    // −∂Ex/∂y
    let ex_yp = model_ex.forward(x.clone(), y.clone().add_scalar(h), t.clone());
    let ex_ym = model_ex.forward(x.clone(), y.clone().sub_scalar(h), t.clone());
    let dex_dy = (ex_yp.sub(ex_ym)).div_scalar(2.0 * h);

    // +∂Ey/∂x
    let ey_xp = model_ey.forward(x.clone().add_scalar(h), y.clone(), t.clone());
    let ey_xm = model_ey.forward(x.clone().sub_scalar(h), y.clone(), t.clone());
    let dey_dx = (ey_xp.sub(ey_xm)).div_scalar(2.0 * h);

    dhz_dt.mul_scalar(mu as f32).sub(dex_dy).add(dey_dx)
}

/// TE-mode Ampère-x residual: R_{Ax} = ε ∂Ex/∂t − ∂Hz/∂y + σ Ex
///
/// ## Theorem — 2D TE Ampère-x
///
/// The x-component of Ampère's law ∇×H = σE + ε ∂E/∂t in 2D TE:
/// ```text
/// R_{Ax} = ε ∂Ex/∂t  −  ∂Hz/∂y  +  σ Ex  =  0
/// ```
/// (No external current source in this component; for impressed J add separately.)
///
/// **Proof.** Substituting Ex = A·e^{i(ky·y − ωt)}, Hz from Faraday's law:
/// Hz = −(A·ky / (ωμ))·e^{i(ky·y − ωt)}, then ∂Hz/∂y = −i·ky·Hz.
/// R_Ax = ε·(−iω)·Ex − (−i·ky)·Hz = ε·(−iω)·A − ky²·A/(ωμ).
/// Setting = 0: εμω² = ky² (dispersion). ✓
///
/// # References — see `te_mode_faraday_residual`.
pub fn te_mode_ampere_x_residual<B: AutodiffBackend>(
    model_ex: &BurnPINN2DWave<B>,
    model_hz: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    t: &Tensor<B, 2>,
    eps: f64,
    sigma: f64,
) -> Tensor<B, 2> {
    let h = EPS_FD_F32;

    // ε ∂Ex/∂t
    let ex_tp = model_ex.forward(x.clone(), y.clone(), t.clone().add_scalar(h));
    let ex_tm = model_ex.forward(x.clone(), y.clone(), t.clone().sub_scalar(h));
    let dex_dt = (ex_tp.sub(ex_tm)).div_scalar(2.0 * h);

    // −∂Hz/∂y
    let hz_yp = model_hz.forward(x.clone(), y.clone().add_scalar(h), t.clone());
    let hz_ym = model_hz.forward(x.clone(), y.clone().sub_scalar(h), t.clone());
    let dhz_dy = (hz_yp.sub(hz_ym)).div_scalar(2.0 * h);

    // σ Ex (conduction current)
    let ex = model_ex.forward(x.clone(), y.clone(), t.clone());

    dex_dt
        .mul_scalar(eps as f32)
        .sub(dhz_dy)
        .add(ex.mul_scalar(sigma as f32))
}

/// TE-mode Ampère-y residual: R_{Ay} = ε ∂Ey/∂t + ∂Hz/∂x + σ Ey
///
/// ## Theorem — 2D TE Ampère-y
///
/// The y-component of Ampère's law in 2D TE:
/// ```text
/// R_{Ay} = ε ∂Ey/∂t  +  ∂Hz/∂x  +  σ Ey  =  0
/// ```
/// Note the **+∂Hz/∂x** sign: in 2D TE, (∇×H)_y = +∂Hz/∂x (no ∂Hx/∂z term).
///
/// # References — see `te_mode_faraday_residual`.
pub fn te_mode_ampere_y_residual<B: AutodiffBackend>(
    model_ey: &BurnPINN2DWave<B>,
    model_hz: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    t: &Tensor<B, 2>,
    eps: f64,
    sigma: f64,
) -> Tensor<B, 2> {
    let h = EPS_FD_F32;

    // ε ∂Ey/∂t
    let ey_tp = model_ey.forward(x.clone(), y.clone(), t.clone().add_scalar(h));
    let ey_tm = model_ey.forward(x.clone(), y.clone(), t.clone().sub_scalar(h));
    let dey_dt = (ey_tp.sub(ey_tm)).div_scalar(2.0 * h);

    // +∂Hz/∂x
    let hz_xp = model_hz.forward(x.clone().add_scalar(h), y.clone(), t.clone());
    let hz_xm = model_hz.forward(x.clone().sub_scalar(h), y.clone(), t.clone());
    let dhz_dx = (hz_xp.sub(hz_xm)).div_scalar(2.0 * h);

    // σ Ey
    let ey = model_ey.forward(x.clone(), y.clone(), t.clone());

    dey_dt
        .mul_scalar(eps as f32)
        .add(dhz_dx)
        .add(ey.mul_scalar(sigma as f32))
}

/// TE-mode Gauss residual: R_G = ε (∂Ex/∂x + ∂Ey/∂y) − ρ_free
///
/// ## Theorem — 2D TE Gauss's Law for E
///
/// In 2D TE, the divergence of the transverse E-field equals free charge density:
/// ```text
/// R_G = ε (∂Ex/∂x  +  ∂Ey/∂y)  −  ρ_free  =  0
/// ```
/// This is the 2D projection of ∇·(εE) = ρ_free.
///
/// **Proof.** Plane wave E = A·(kx̂ + k'ŷ)·e^{i(k·r−ωt)}: ∇·E = i·A·(k+k')·e^{...},
/// which is zero if the wave has no free charges. ✓
///
/// # References
/// - Jackson, J.D. (1999). *Classical Electrodynamics* (3rd ed.). §6.1.
pub fn te_mode_gauss_residual<B: AutodiffBackend>(
    model_ex: &BurnPINN2DWave<B>,
    model_ey: &BurnPINN2DWave<B>,
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    t: &Tensor<B, 2>,
    eps: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Tensor<B, 2> {
    let h = EPS_FD_F32;

    // ∂Ex/∂x
    let ex_xp = model_ex.forward(x.clone().add_scalar(h), y.clone(), t.clone());
    let ex_xm = model_ex.forward(x.clone().sub_scalar(h), y.clone(), t.clone());
    let dex_dx = (ex_xp.sub(ex_xm)).div_scalar(2.0 * h);

    // ∂Ey/∂y
    let ey_yp = model_ey.forward(x.clone(), y.clone().add_scalar(h), t.clone());
    let ey_ym = model_ey.forward(x.clone(), y.clone().sub_scalar(h), t.clone());
    let dey_dy = (ey_yp.sub(ey_ym)).div_scalar(2.0 * h);

    let rho = compute_charge_density(x, y, physics_params);

    dex_dx.add(dey_dy).mul_scalar(eps as f32).sub(rho)
}
