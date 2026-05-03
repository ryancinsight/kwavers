//! Test C — Yoshida4 convergence order via global trajectory error.

use super::super::{YOSHIDA_W1, YOSHIDA_W2};
use std::f64::consts::PI;

/// **Test C — Yoshida4 convergence order via global trajectory error.**
///
/// The Rayleigh-Plesset equation has a velocity-dependent force term (−3Ṙ²/2R),
/// making it a non-separable Hamiltonian system. For such systems, the energy
/// oscillation amplitude scales as O(h) per period (not O(h²) as for separable H),
/// so H_spread is NOT an appropriate convergence metric.
///
/// Instead we use the **global trajectory error** ε(h) = |R(t_end; h) − R_ref| / R₀,
/// where R_ref is a high-accuracy Y4 reference at N_ref = 2000 steps/period.
/// For any consistent p-th order one-step method, ε(h) ~ C h^p t_end, giving:
///
///   Störmer-Verlet (p = 2): slope = 2.0
///   Yoshida4       (p = 4): slope = 4.0
///
/// Measurement: run on a SEPARABLE simple harmonic oscillator (SHO):
///   d²x/dt² = −ω²x   (force depends only on position, not velocity)
///
/// This is the correct test case for Yoshida4, because:
///   1. The Keller-Miksis/Rayleigh-Plesset equation has a velocity-dependent
///      force term −3Ṙ²/(2R), making it NON-SEPARABLE.  For non-separable systems
///      the velocity-Verlet base step is not time-reversible (Φ_{−h} ≠ Φ_h^{−1}),
///      so the Yoshida BCH error cancellation does not produce O(h⁴) — it gives O(h²)
///      at best.  (Hairer et al. 2006, §V.3.1: composition requires a *symmetric*
///      base method.)
///   2. For the SHO, f = −ω²x depends only on position → velocity Verlet is
///      exactly time-reversible → Yoshida composition gives O(h⁴). ✓
///
/// SHO exact solution: x(t) = x₀ cos(ωt),  v(t) = −x₀ ω sin(ωt).
/// Integrate for t_end = 10π (5 full periods) and compare to the exact value.
///
/// References:
/// - Hairer, Lubich & Wanner (2006) *Geometric Numerical Integration* §II.1, §V.3.
/// - Yoshida (1990) Phys. Lett. A 150:262.
#[test]
fn test_yoshida4_order() {
    // SHO parameters: ω = 1, x₀ = 1, v₀ = 0
    // t_end is a NON-INTEGER multiple of T = 2π to avoid super-convergence.
    // At exactly integer periods, SV phase error cancels (cos(ω̃·nT)≈cos(ωnT)=1)
    // giving spurious O(h⁴) for SV and O(h⁸) for Y4 — 2× the true order.
    // Using t_end = 5.7·T ensures no such cancellation.
    let omega = 1.0_f64;
    let x0 = 1.0_f64;
    let t_end = 5.7 * 2.0 * PI; // 5.7 periods (non-integer → no super-convergence)

    // Exact solution at t_end: x = cos(ω·t_end)
    let x_exact = (omega * t_end).cos();

    // Force for SHO: f(x) = −ω²·x  (position-only → separable)
    let f_sho = |x: f64| -> f64 { -omega * omega * x };

    // Störmer-Verlet step for SHO: half-kick / drift / half-kick
    // V_{n+½} = V_n + (h/2)·f(X_n)
    // X_{n+1} = X_n + h·V_{n+½}
    // V_{n+1} = V_{n+½} + (h/2)·f(X_{n+1})
    let sv_step_sho = |x: &mut f64, v: &mut f64, dt: f64| {
        let a0 = f_sho(*x);
        *v += 0.5 * dt * a0;
        *x += dt * *v;
        let a1 = f_sho(*x);
        *v += 0.5 * dt * a1;
    };

    // Yoshida4 step: Ψ_h = Φ_{w₁h} ∘ Φ_{w₂h} ∘ Φ_{w₁h}
    // For the separable SHO, this is symplectic and gives O(h⁴).
    let y4_step_sho = |x: &mut f64, v: &mut f64, dt: f64| {
        sv_step_sho(x, v, YOSHIDA_W1 * dt);
        sv_step_sho(x, v, YOSHIDA_W2 * dt);
        sv_step_sho(x, v, YOSHIDA_W1 * dt);
    };

    // Run N steps over [0, t_end], return x(t_end).
    let run_sho = |n: usize, use_y4: bool| -> f64 {
        let dt = t_end / n as f64;
        let mut x = x0;
        let mut v = 0.0_f64;
        for _ in 0..n {
            if use_y4 {
                y4_step_sho(&mut x, &mut v, dt);
            } else {
                sv_step_sho(&mut x, &mut v, dt);
            }
        }
        x
    };

    // Compare at N = 100, 200, 400 steps over t_end (≈ 6.4, 3.2, 1.6 steps/period).
    // Outer pair gives dt ratio 1/4 → slope = ln(err_400/err_100)/ln(0.25).
    let ns = [100_usize, 200, 400];
    let mut sv_errs: Vec<f64> = Vec::new();
    let mut y4_errs: Vec<f64> = Vec::new();

    for &n in &ns {
        sv_errs.push((run_sho(n, false) - x_exact).abs());
        y4_errs.push((run_sho(n, true) - x_exact).abs());
    }

    println!(
        "SHO SV  errors at N=100/200/400: {:.4e} / {:.4e} / {:.4e}",
        sv_errs[0], sv_errs[1], sv_errs[2]
    );
    println!(
        "SHO Y4  errors at N=100/200/400: {:.4e} / {:.4e} / {:.4e}",
        y4_errs[0], y4_errs[1], y4_errs[2]
    );

    // Log-log slope: slope = ln(err(N₃)/err(N₁)) / ln(N₁/N₃) = ln(ratio)/ln(0.25)
    let log_n_ratio = (100.0_f64 / 400.0_f64).ln(); // ln(0.25) ≈ −1.386
    let sv_slope = (sv_errs[2] / sv_errs[0].max(1e-30)).ln() / log_n_ratio;
    let y4_slope = (y4_errs[2] / y4_errs[0].max(1e-30)).ln() / log_n_ratio;

    println!(
        "Convergence slopes — SV: {:.2} (expected 2.0), Y4: {:.2} (expected 4.0)",
        sv_slope, y4_slope
    );

    // Tolerance ±30% of expected order
    assert!(
        (sv_slope - 2.0).abs() / 2.0 < 0.30,
        "Störmer-Verlet SHO slope {:.2} deviates >30% from 2.0 \
         (SV errors: {:?})",
        sv_slope,
        sv_errs
    );
    assert!(
        (y4_slope - 4.0).abs() / 4.0 < 0.30,
        "Yoshida4 SHO slope {:.2} deviates >30% from 4.0 \
         (Y4 errors: {:?})",
        y4_slope,
        y4_errs
    );
}
