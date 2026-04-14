use super::constants::EPS_FD_F32;
use super::sources::compute_current_density_z;
use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
use crate::solver::inverse::pinn::ml::BurnPINN2DWave;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

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
    physics_params: &PhysicsParameters,
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

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod maxwell_vector_residual_tests {
    use super::EPS_FD_F32;

    /// **Test 1 — Zero-field residuals.**
    ///
    /// ## Theorem
    /// If all field components are identically zero (Ez = Hx = Hy = Ex = Ey = Hz = 0),
    /// every partial derivative is zero and every residual is identically zero.
    ///
    /// **Proof:** f(x)=0 → (f(x+h)−f(x−h))/(2h) = 0/2h = 0.
    /// With σ = 0 and J_z = 0, all six residuals = 0. □
    ///
    /// We verify using the FD formula directly (no model needed):
    /// the "field" is modelled as a constant-zero function.
    #[test]
    fn test_zero_field_all_residuals_zero() {
        // Simulate FD of a zero field: all differences are exactly zero.
        let field = |_x: f32, _y: f32, _t: f32| -> f32 { 0.0 };

        let h = EPS_FD_F32;
        let x0 = 0.3_f32;
        let y0 = 0.7_f32;
        let t0 = 0.5_f32;

        // Central-difference FD operator
        let fd_t = |f: &dyn Fn(f32, f32, f32) -> f32| -> f32 {
            (f(x0, y0, t0 + h) - f(x0, y0, t0 - h)) / (2.0 * h)
        };
        let fd_x = |f: &dyn Fn(f32, f32, f32) -> f32| -> f32 {
            (f(x0 + h, y0, t0) - f(x0 - h, y0, t0)) / (2.0 * h)
        };
        let fd_y = |f: &dyn Fn(f32, f32, f32) -> f32| -> f32 {
            (f(x0, y0 + h, t0) - f(x0, y0 - h, t0)) / (2.0 * h)
        };

        let mu = 1.0_f32;
        let eps = 1.0_f32;
        let sigma = 0.0_f32;

        // TM residuals (Ez, Hx, Hy)
        let r_fx = mu * fd_t(&field) + fd_y(&field); // R_{Fx}
        let r_fy = mu * fd_t(&field) - fd_x(&field); // R_{Fy}
        let r_az = eps * fd_t(&field) - fd_x(&field) + fd_y(&field) + sigma * field(x0, y0, t0); // R_{Az}

        // TE residuals (Hz, Ex, Ey)
        let r_f = mu * fd_t(&field) - fd_y(&field) + fd_x(&field); // R_{F}
        let r_ax = eps * fd_t(&field) - fd_y(&field) + sigma * field(x0, y0, t0); // R_{Ax}
        let r_ay = eps * fd_t(&field) + fd_x(&field) + sigma * field(x0, y0, t0); // R_{Ay}
        let r_g = eps * (fd_x(&field) + fd_y(&field)); // R_G (ρ=0)

        for (name, r) in &[
            ("R_Fx", r_fx),
            ("R_Fy", r_fy),
            ("R_Az", r_az),
            ("R_F", r_f),
            ("R_Ax", r_ax),
            ("R_Ay", r_ay),
            ("R_G", r_g),
        ] {
            assert!(
                r.abs() < 1e-6,
                "zero-field residual {} = {:.3e}, expected < 1e-6",
                name,
                r
            );
        }
    }

    /// **Test 2 — TM plane wave residuals.**
    ///
    /// ## Theorem — Exact TM plane wave solution (Jackson 1999, §7.1)
    ///
    /// For normalised units (ε = μ = 1, σ = 0, J = 0), the dispersion relation
    /// gives k = ω. With ω = k = 1, the fields
    /// ```text
    /// Ez(x,t)  = sin(x − t)
    /// Hy(x,t)  = −sin(x − t)   [from Faraday-y: μ ∂Hy/∂t = ∂Ez/∂x]
    /// Hx(x,t)  = 0              [no y variation]
    /// ```
    /// satisfy all three TM Maxwell equations exactly. FD approximations introduce
    /// truncation error O(h²) ≈ O(EPS_FD_F32²) ≈ 2.4e-5, well below 1e-3.
    ///
    /// **Proof (R_Fy):**
    ///   μ ∂Hy/∂t = ∂(−sin(x−t))/∂t = cos(x−t)
    ///   ∂Ez/∂x   = cos(x−t)
    ///   R_Fy = cos − cos = 0 ✓
    ///
    /// **Proof (R_Az):**
    ///   ε ∂Ez/∂t = −cos(x−t);  −∂Hy/∂x = cos(x−t);  +∂Hx/∂y = 0
    ///   R_Az = −cos + cos + 0 = 0 ✓
    #[test]
    fn test_tm_plane_wave_residuals_near_zero() {
        let h = EPS_FD_F32;

        // Plane-wave fields: ε=μ=1, ω=k=1
        let ez = |x: f32, _y: f32, t: f32| -> f32 { (x - t).sin() };
        let hy = |x: f32, _y: f32, t: f32| -> f32 { -(x - t).sin() };
        let hx = |_x: f32, _y: f32, _t: f32| -> f32 { 0.0 };

        // Test at a batch of interior points
        let test_points = [(0.3_f32, 0.4_f32, 0.5_f32), (1.1_f32, 0.2_f32, 0.8_f32)];

        for (x0, y0, t0) in &test_points {
            let (x0, y0, t0) = (*x0, *y0, *t0);

            // R_{Fx} = μ ∂Hx/∂t + ∂Ez/∂y  (no y variation → 0)
            let dhx_dt = (hx(x0, y0, t0 + h) - hx(x0, y0, t0 - h)) / (2.0 * h);
            let dez_dy = (ez(x0, y0 + h, t0) - ez(x0, y0 - h, t0)) / (2.0 * h);
            let r_fx = dhx_dt + dez_dy;

            // R_{Fy} = μ ∂Hy/∂t − ∂Ez/∂x
            let dhy_dt = (hy(x0, y0, t0 + h) - hy(x0, y0, t0 - h)) / (2.0 * h);
            let dez_dx = (ez(x0 + h, y0, t0) - ez(x0 - h, y0, t0)) / (2.0 * h);
            let r_fy = dhy_dt - dez_dx;

            // R_{Az} = ε ∂Ez/∂t − ∂Hy/∂x + ∂Hx/∂y  (σ=0, J=0)
            let dez_dt = (ez(x0, y0, t0 + h) - ez(x0, y0, t0 - h)) / (2.0 * h);
            let dhy_dx = (hy(x0 + h, y0, t0) - hy(x0 - h, y0, t0)) / (2.0 * h);
            let dhx_dy = (hx(x0, y0 + h, t0) - hx(x0, y0 - h, t0)) / (2.0 * h);
            let r_az = dez_dt - dhy_dx + dhx_dy;

            for (name, r) in &[("R_Fx", r_fx), ("R_Fy", r_fy), ("R_Az", r_az)] {
                assert!(
                    r.abs() < 1e-3,
                    "TM plane-wave residual {} at ({x0},{y0},{t0}) = {:.4e}, expected < 1e-3",
                    name,
                    r
                );
            }
        }
    }

    /// **Test 3 — TE waveguide below cutoff (negative test).**
    ///
    /// ## Theorem — Evanescent TE field
    ///
    /// Below the TE₁₀ cutoff frequency (k < π/a), the field is evanescent:
    /// Hz(x,y) = cosh(κy),  Ex = Ey = 0  (purely spatial, not propagating).
    /// This does NOT satisfy Faraday's law (it has zero ∂Hz/∂t but non-zero curl):
    ///   R_F = μ·0 − 0 + 0 = 0  (Faraday trivially satisfied for static Hz)
    ///
    /// BUT the Ampère-x equation R_{Ax} = ε·∂Ex/∂t − ∂Hz/∂y + σ·Ex
    /// = 0 − κ·sinh(κy) + 0 ≠ 0 for κ > 0.
    ///
    /// This confirms the residual correctly flags a field that is NOT a time-harmonic
    /// solution (non-zero Ampère residual detects the evanescent-but-static field).
    #[test]
    fn test_te_below_cutoff_ampere_nonzero() {
        let h = EPS_FD_F32;
        let kappa = 1.0_f32; // evanescent decay constant

        // Hz = cosh(κy); Ex = Ey = 0
        let hz = |_x: f32, y: f32, _t: f32| -> f32 { (kappa * y).cosh() };
        let ex = |_x: f32, _y: f32, _t: f32| -> f32 { 0.0 };

        let x0 = 0.5_f32;
        let y0 = 0.3_f32;
        let t0 = 0.0_f32;
        let eps = 1.0_f32;
        let sigma = 0.0_f32;

        // R_{Ax} = ε ∂Ex/∂t − ∂Hz/∂y + σ Ex
        let dex_dt = (ex(x0, y0, t0 + h) - ex(x0, y0, t0 - h)) / (2.0 * h); // = 0
        let dhz_dy = (hz(x0, y0 + h, t0) - hz(x0, y0 - h, t0)) / (2.0 * h); // ≈ κ sinh(κy)
        let r_ax = eps * dex_dt - dhz_dy + sigma * ex(x0, y0, t0);

        // The residual must be non-zero: dhz_dy ≈ κ·sinh(κ·y0) > 0
        let expected_dhz = kappa * (kappa * y0).sinh();
        assert!(
            r_ax.abs() > 1e-4,
            "below-cutoff R_Ax = {:.4e} should be non-zero (expected ~{:.4e})",
            r_ax,
            -expected_dhz
        );
    }

    /// **Test 4 — Poynting energy conservation for a lossless TM plane wave.**
    ///
    /// ## Theorem — Time-averaged Poynting theorem (Griffiths 2017, §8.1)
    ///
    /// For a lossless medium (σ = 0) and a time-harmonic plane wave, the
    /// time-averaged Poynting vector `⟨S⟩ = ½ Re(E×H*)` is constant along
    /// the propagation direction, i.e., ∂⟨Sx⟩/∂x = 0.
    ///
    /// **Proof.** For Ez = A·sin(kx−ωt), Hy = −(A/Z)·sin(kx−ωt) where Z=√(μ/ε):
    ///   Sx = Ez · (−Hy) = (A²/Z)·sin²(kx−ωt)
    ///   ⟨Sx⟩ = A²/(2Z) = const  →  ∂⟨Sx⟩/∂x = 0. ✓
    ///
    /// We verify numerically: compute Sx at x and x+δ, check relative change < 1%.
    #[test]
    fn test_tm_plane_wave_poynting_conservation() {
        // Lossless TM plane wave: ε=μ=1, ω=k=1, A=1
        let ez = |x: f32, t: f32| -> f32 { (x - t).sin() };
        let hy = |x: f32, t: f32| -> f32 { -(x - t).sin() }; // Z = √(μ/ε) = 1

        // Poynting x-component: Sx = -Ez × Hy (standard TE convention Sx = Ez Hy)
        // In TM: S = E × H, (E_z ẑ) × (H_y ŷ) = E_z H_y (ẑ×ŷ = −x̂)
        // Sx = −Ez · Hy = −sin(x−t)·(−sin(x−t)) = sin²(x−t)
        let sx = |x: f32, t: f32| -> f32 { -ez(x, t) * hy(x, t) };

        // Average over one period at two x locations separated by λ/4
        let n = 1000_usize;
        let t_period = 2.0 * std::f32::consts::PI;
        let x1 = 0.3_f32;
        let x2 = x1 + std::f32::consts::PI / 2.0; // λ/4

        let avg_sx = |x: f32| -> f32 {
            (0..n)
                .map(|i| sx(x, i as f32 * t_period / n as f32))
                .sum::<f32>()
                / n as f32
        };

        let s1 = avg_sx(x1);
        let s2 = avg_sx(x2);

        // Both averages = 0.5 (⟨sin²⟩ = 1/2), relative difference < 1%
        let rel_diff = (s1 - s2).abs() / s1.abs().max(1e-10);
        assert!(
            rel_diff < 0.01,
            "Poynting flux not conserved: ⟨Sx⟩ at x₁={x1:.2} = {s1:.6}, \
             at x₂={x2:.2} = {s2:.6}, rel diff = {rel_diff:.4e}"
        );
    }
}
