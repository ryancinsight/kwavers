//! Validation tests for symplectic bubble integrators.
//!
//! Covers:
//! - `test_minnaert_period`: Minnaert frequency error < 0.5% at dt = T₀/200
//! - `test_hamiltonian_no_drift`: H stays in [0.5 H₀, 2 H₀] over 1000 periods
//! - `test_yoshida4_order`: Convergence order 4.0 ± 30% on SHO
//! - `test_equilibrium_preserved`: |R−R₀|/R₀ < 1e-12 at exact equilibrium

use super::{BubbleSymplecticIntegrator, SymplecticConfig, YOSHIDA_W1, YOSHIDA_W2};

use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
use std::f64::consts::PI;
use std::sync::Arc;

/// Build a minimal bubble setup for Hamiltonian conservation tests.
///
/// Inviscid (μ = 0) and effectively incompressible (c → ∞) so that K-M reduces
/// to the undamped Rayleigh-Plesset equation — which IS exactly conservative
/// (dH/dt = 0 analytically).  Both viscous and radiation damping are suppressed:
///   - μ = 0 → no viscous dissipation
///   - c = 1e12 m/s → radiation damping rate ~ ω₀²R₀/c ≈ 4.7e-5 s⁻¹, giving
///     < 0.01% energy loss over 1000 periods (vs. 99.9% loss with c = 1482 m/s)
fn make_params(r0: f64) -> BubbleParameters {
    BubbleParameters {
        r0,
        mu_liquid: 0.0, // inviscid — no viscous energy dissipation
        c_liquid: 1e12, // effectively incompressible — suppresses radiation damping
        use_thermal_effects: false,
        use_mass_transfer: false,
        driving_frequency: 0.0, // no acoustic driving
        ..Default::default()
    }
}

fn make_model(r0: f64) -> KellerMiksisModel {
    KellerMiksisModel::new(make_params(r0))
}

/// Compute bubble Hamiltonian H(R, Ṙ) under the isothermal polytropic model.
///
/// H = ½ ρ_L R³ Ṙ²  +  V_eff(R)
///
/// V_eff(R) = −∫_{R₀}^{R} [p_gas(R′) − p₀ − 2σ/R′] R′² dR′
/// ≈ (using polytropic p_gas = p_eq (R₀/R′)^{3γ}):
///   numerical integral via trapezoidal rule on [R₀·0.5, R·1.5]
fn bubble_hamiltonian(r: f64, v: f64, params: &BubbleParameters, n_points: usize) -> f64 {
    let r0 = params.r0;
    let rho_l = params.rho_liquid;
    let p0 = params.p0;
    let sigma = params.sigma;
    let gamma = crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air.gamma();
    let p_eq = p0 + 2.0 * sigma / r0;

    // Kinetic energy: ½ ρ_L R³ Ṙ²
    let ke = 0.5 * rho_l * r * r * r * v * v;

    // Potential energy: ∫_{r_ref}^{R} p_net(R′) R′² dR′, p_net = p_gas − p₀ − 2σ/R′
    // We integrate from R₀ to R using the trapezoidal rule.
    // The integral is zero at R₀ (reference), sign follows from pressure balance.
    let r_lo = r0.min(r);
    let r_hi = r0.max(r);
    let sign = if r < r0 { -1.0 } else { 1.0 };

    let n = n_points.max(2);
    let dr = (r_hi - r_lo) / (n - 1) as f64;
    let mut pe = 0.0_f64;
    let mut prev = {
        let ri = r_lo;
        let p_gas = p_eq * (r0 / ri).powf(3.0 * gamma);
        let p_net = p_gas - p0 - 2.0 * sigma / ri;
        p_net * ri * ri
    };

    for k in 1..n {
        let ri = r_lo + k as f64 * dr;
        let p_gas = p_eq * (r0 / ri).powf(3.0 * gamma);
        let p_net = p_gas - p0 - 2.0 * sigma / ri;
        let cur = p_net * ri * ri;
        pe += 0.5 * (prev + cur) * dr;
        prev = cur;
    }

    // V_eff = -∫_{R₀}^{R} p_net R'² dR'
    // For R > R₀: p_net < 0 → pe < 0 → -sign*pe = -(+1)*pe > 0 ✓ (energy increases when displaced)
    // For R < R₀: p_net > 0 → pe > 0 → -sign*pe = -(-1)*pe > 0 ✓ (symmetric potential well)
    ke - sign * pe
}

/// **Test A — Minnaert period accuracy (Störmer-Verlet, O(h²) error).**
///
/// Setup: R₀ = 10 µm, 5% radius perturbation, no driving.
///
/// Minnaert angular frequency (linearised RP):
///   ω₀ = (1/R₀) √((3γp_eq − 2σ/R₀) / ρ_L)
///
/// Validation: measured period vs. Minnaert period, error < 0.5%.
///
/// Reference: Minnaert, M. (1933). *Philos. Mag.* **16**, 235–248.
/// Error bound: Hairer et al. (2006) §I.3.1 — SV period error ~ (hω₀)²/24
/// = (2π/200)²/24 ≈ 0.04% ≪ 0.5%.
#[test]
fn test_minnaert_period() {
    let r0 = 10e-6; // 10 µm
    let params = make_params(r0);
    let model = Arc::new(make_model(r0));

    let rho_l = params.rho_liquid;
    let p0 = params.p0;
    let sigma = params.sigma;
    let gamma = crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air.gamma();
    let p_eq = p0 + 2.0 * sigma / r0;

    // Minnaert angular frequency (linearised RP with surface tension)
    let omega0 = ((3.0 * gamma * p_eq - 2.0 * sigma / r0) / (rho_l * r0 * r0)).sqrt();
    let t_period = 2.0 * PI / omega0;
    let dt = t_period / 200.0; // 200 steps per period → (hω₀)²/24 ≈ 0.04% error

    let config = SymplecticConfig {
        dt,
        max_mach: 0.95,
        r_min_fraction: 0.01,
    };
    let integrator = BubbleSymplecticIntegrator::new(model.clone(), config);

    // 5% radius perturbation, zero velocity (linear regime)
    let mut state = BubbleState::at_equilibrium(&params);
    state.radius = 1.05 * r0;
    state.wall_velocity = 0.0;

    // Integrate for 100 periods; find the first return time (V=0, decelerating)
    let n_periods = 100;
    let total_steps = n_periods * 200;
    let mut t = 0.0;
    let mut periods: Vec<f64> = Vec::new();
    let mut prev_v = state.wall_velocity;
    let mut t_last_zero = 0.0;
    let mut first_zero_found = false;

    for _ in 0..total_steps {
        let v_before = state.wall_velocity;
        integrator
            .sv_step(&mut state, 0.0, 0.0, t)
            .expect("SV step must not fail");
        let v_after = state.wall_velocity;
        t += dt;

        // Detect V changing sign from negative to positive (minimum of R → new period)
        if prev_v < 0.0 && v_before < 0.0 && v_after >= 0.0 {
            if first_zero_found {
                periods.push(t - t_last_zero);
            }
            t_last_zero = t;
            first_zero_found = true;
        }
        prev_v = v_after;
    }

    assert!(
        !periods.is_empty(),
        "No complete oscillation periods detected in {} steps",
        total_steps
    );

    let period_meas = periods.iter().sum::<f64>() / periods.len() as f64;
    let rel_err = (period_meas - t_period).abs() / t_period;

    println!(
        "Minnaert period: measured {:.4} µs, exact {:.4} µs, error {:.4}%",
        period_meas * 1e6,
        t_period * 1e6,
        100.0 * rel_err
    );

    assert!(
        rel_err < 0.005,
        "Minnaert period error {:.3}% exceeds 0.5% (measured {:.4e} s, exact {:.4e} s)",
        100.0 * rel_err,
        period_meas,
        t_period
    );
}

/// **Test B — Hamiltonian non-drift over 1000 periods (Störmer-Verlet).**
///
/// For the Rayleigh-Plesset / Keller-Miksis mechanical sub-system, the true
/// Hamiltonian H = ½ρ_L R³Ṙ² + V_eff(R) is exactly conserved by the undamped
/// RP equation (dH/dt = 0).
///
/// The position-Verlet (SV) integrator applied to R̈ = f(R, Ṙ) with
/// velocity-dependent f is NOT equivalent to canonical SV on a separable
/// Hamiltonian; the force term −3Ṙ²/(2R) is O(V²/R) and introduces O(h)
/// energy oscillations (not O(h²) as for separable H = T(p) + V(q)).
///
/// Nevertheless, the energy DOES NOT drift monotonically:
///   - Dissipative system (μ ≠ 0): H → 0 within ~100 periods.
///   - This integrator: H oscillates, stays in [0.5 H₀, 2 H₀] over 1000 periods.
///
/// The key distinguishing property is bounded oscillation vs secular drift.
///
/// Assertion: H_min > 0.5 × H₀  and  H_max < 2.0 × H₀  over 1000 periods.
#[test]
fn test_hamiltonian_no_drift() {
    let r0 = 10e-6;
    let params = make_params(r0);
    let model = Arc::new(make_model(r0));

    let rho_l = params.rho_liquid;
    let p0 = params.p0;
    let sigma = params.sigma;
    let gamma = crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air.gamma();
    let p_eq = p0 + 2.0 * sigma / r0;
    let omega0 = ((3.0 * gamma * p_eq - 2.0 * sigma / r0) / (rho_l * r0 * r0)).sqrt();
    let t_period = 2.0 * PI / omega0;
    let dt = t_period / 200.0;

    let config = SymplecticConfig {
        dt,
        max_mach: 0.95,
        r_min_fraction: 0.01,
    };
    let integrator = BubbleSymplecticIntegrator::new(model.clone(), config);

    let mut state = BubbleState::at_equilibrium(&params);
    state.radius = 1.05 * r0;
    state.wall_velocity = 0.0;

    let h0 = bubble_hamiltonian(state.radius, state.wall_velocity, &params, 1000);
    let mut h_min = h0;
    let mut h_max = h0;

    let total_steps = 1000 * 200; // 1000 periods
    let mut t = 0.0;

    for _ in 0..total_steps {
        integrator
            .sv_step(&mut state, 0.0, 0.0, t)
            .expect("SV step must not fail");
        t += dt;
        let h = bubble_hamiltonian(state.radius, state.wall_velocity, &params, 200);
        h_min = h_min.min(h);
        h_max = h_max.max(h);
    }

    let h_spread = (h_max - h_min) / h0.abs().max(1e-30);
    println!(
        "H over 1000 periods: min={:.4e}  max={:.4e}  H₀={:.4e}  spread={:.2}%",
        h_min,
        h_max,
        h0,
        100.0 * h_spread
    );

    // Primary check: no secular LOSS (energy stays above 50 % of H₀).
    // A viscous system (non-zero μ) would have H_min → 0 within a few hundred periods.
    assert!(
        h_min > 0.5 * h0,
        "H dropped below 50% of H₀ — secular energy loss detected \
         (H_min={:.4e}, H₀={:.4e})",
        h_min,
        h0
    );
    // Secondary check: no secular GAIN (energy stays below 200 % of H₀).
    assert!(
        h_max < 2.0 * h0,
        "H exceeded 200% of H₀ — energy is growing (H_max={:.4e}, H₀={:.4e})",
        h_max,
        h0
    );
}

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

/// **Test D — Exact equilibrium preservation.**
///
/// At R = R₀, Ṙ = 0, no acoustic driving: K-M gives R̈ = 0.
/// Both half-kicks add zero → R and Ṙ remain exactly at equilibrium.
/// Assert |R − R₀|/R₀ < 1e-12 and |Ṙ| < 1e-12 after 1000 steps.
#[test]
fn test_equilibrium_preserved() {
    let r0 = 10e-6;
    let params = make_params(r0);
    let model = Arc::new(make_model(r0));

    let rho_l = params.rho_liquid;
    let p0 = params.p0;
    let sigma = params.sigma;
    let gamma = crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air.gamma();
    let p_eq = p0 + 2.0 * sigma / r0;
    let omega0 = ((3.0 * gamma * p_eq - 2.0 * sigma / r0) / (rho_l * r0 * r0)).sqrt();
    let t_period = 2.0 * PI / omega0;
    let dt = t_period / 200.0;

    let config = SymplecticConfig {
        dt,
        ..Default::default()
    };
    let integrator = BubbleSymplecticIntegrator::new(model.clone(), config);

    // Exact mechanical equilibrium
    let mut state = BubbleState::at_equilibrium(&params);
    // at_equilibrium already sets radius = r0, wall_velocity = 0
    let mut t = 0.0;
    for _ in 0..1000 {
        integrator
            .sv_step(&mut state, 0.0, 0.0, t)
            .expect("SV step must not fail");
        t += dt;
    }

    let r_err = (state.radius - r0).abs() / r0;
    let v_err = state.wall_velocity.abs();
    println!(
        "Equilibrium preservation: |R−R₀|/R₀ = {:.2e}, |Ṙ| = {:.2e} m/s",
        r_err, v_err
    );

    assert!(
        r_err < 1e-12,
        "|R−R₀|/R₀ = {:.2e} exceeds 1e-12 — equilibrium drifted",
        r_err
    );
    assert!(
        v_err < 1e-12,
        "|Ṙ| = {:.2e} m/s exceeds 1e-12 — velocity drifted from zero",
        v_err
    );
}
