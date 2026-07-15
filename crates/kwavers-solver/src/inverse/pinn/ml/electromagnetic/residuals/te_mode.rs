use super::constants::EPS_FD_F32;
use super::sources::compute_charge_density;
use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use crate::inverse::pinn::ml::PinnWave2D;
use coeus_autograd::{add, scalar_add, scalar_mul, scalar_sub, sub, Var};

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
pub fn te_mode_faraday_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model_ex: &PinnWave2D<B>,
    model_ey: &PinnWave2D<B>,
    model_hz: &PinnWave2D<B>,
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

    // μ ∂Hz/∂t
    let hz_tp = model_hz.forward(x, y, &scalar_add(t, h));
    let hz_tm = model_hz.forward(x, y, &scalar_sub(t, h));
    let dhz_dt = scalar_mul(&sub(&hz_tp, &hz_tm), 1.0 / (2.0 * h));

    // −∂Ex/∂y
    let ex_yp = model_ex.forward(x, &scalar_add(y, h), t);
    let ex_ym = model_ex.forward(x, &scalar_sub(y, h), t);
    let dex_dy = scalar_mul(&sub(&ex_yp, &ex_ym), 1.0 / (2.0 * h));

    // +∂Ey/∂x
    let ey_xp = model_ey.forward(&scalar_add(x, h), y, t);
    let ey_xm = model_ey.forward(&scalar_sub(x, h), y, t);
    let dey_dx = scalar_mul(&sub(&ey_xp, &ey_xm), 1.0 / (2.0 * h));

    add(&sub(&scalar_mul(&dhz_dt, mu as f32), &dex_dy), &dey_dx)
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
pub fn te_mode_ampere_x_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model_ex: &PinnWave2D<B>,
    model_hz: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    t: &Var<f32, B>,
    eps: f64,
    sigma: f64,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let h = EPS_FD_F32;

    // ε ∂Ex/∂t
    let ex_tp = model_ex.forward(x, y, &scalar_add(t, h));
    let ex_tm = model_ex.forward(x, y, &scalar_sub(t, h));
    let dex_dt = scalar_mul(&sub(&ex_tp, &ex_tm), 1.0 / (2.0 * h));

    // −∂Hz/∂y
    let hz_yp = model_hz.forward(x, &scalar_add(y, h), t);
    let hz_ym = model_hz.forward(x, &scalar_sub(y, h), t);
    let dhz_dy = scalar_mul(&sub(&hz_yp, &hz_ym), 1.0 / (2.0 * h));

    // σ Ex (conduction current)
    let ex = model_ex.forward(x, y, t);

    add(
        &sub(&scalar_mul(&dex_dt, eps as f32), &dhz_dy),
        &scalar_mul(&ex, sigma as f32),
    )
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
pub fn te_mode_ampere_y_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model_ey: &PinnWave2D<B>,
    model_hz: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    t: &Var<f32, B>,
    eps: f64,
    sigma: f64,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let h = EPS_FD_F32;

    // ε ∂Ey/∂t
    let ey_tp = model_ey.forward(x, y, &scalar_add(t, h));
    let ey_tm = model_ey.forward(x, y, &scalar_sub(t, h));
    let dey_dt = scalar_mul(&sub(&ey_tp, &ey_tm), 1.0 / (2.0 * h));

    // +∂Hz/∂x
    let hz_xp = model_hz.forward(&scalar_add(x, h), y, t);
    let hz_xm = model_hz.forward(&scalar_sub(x, h), y, t);
    let dhz_dx = scalar_mul(&sub(&hz_xp, &hz_xm), 1.0 / (2.0 * h));

    // σ Ey
    let ey = model_ey.forward(x, y, t);

    add(
        &add(&scalar_mul(&dey_dt, eps as f32), &dhz_dx),
        &scalar_mul(&ey, sigma as f32),
    )
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
pub fn te_mode_gauss_residual<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    model_ex: &PinnWave2D<B>,
    model_ey: &PinnWave2D<B>,
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    t: &Var<f32, B>,
    eps: f64,
    physics_params: &PinnDomainPhysicsParameters,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let h = EPS_FD_F32;

    // ∂Ex/∂x
    let ex_xp = model_ex.forward(&scalar_add(x, h), y, t);
    let ex_xm = model_ex.forward(&scalar_sub(x, h), y, t);
    let dex_dx = scalar_mul(&sub(&ex_xp, &ex_xm), 1.0 / (2.0 * h));

    // ∂Ey/∂y
    let ey_yp = model_ey.forward(x, &scalar_add(y, h), t);
    let ey_ym = model_ey.forward(x, &scalar_sub(y, h), t);
    let dey_dy = scalar_mul(&sub(&ey_yp, &ey_ym), 1.0 / (2.0 * h));

    let rho = compute_charge_density(x, y, physics_params);

    sub(&scalar_mul(&add(&dex_dx, &dey_dy), eps as f32), &rho)
}
